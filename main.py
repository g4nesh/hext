import itertools as it
import torch
from datasets import load_dataset, Image
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms, models

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- knobs ---
heldout = "Colon"         # try "Breast" or "Lung" if val stays empty
j_min = 0.45
img_size = 224
batch_size = 128          # A100-friendly
train_n = 4000
val_n   = 800
max_scan_train = 200_000  # how many samples we scan to find matches
max_scan_val   = 800_000  # scan deeper for val to avoid empties
epochs = 1
lr = 5e-4

GROUPS = {"Immune":[1,2,3], "Stromal":[4,5], "Epithelial":[6], "Melanocyte":[8], "Other":[7,9]}
id2group = {i:g for g,ids in GROUPS.items() for i in ids}
labels   = {g:i for i,g in enumerate(GROUPS.keys())}

# 1) stream
ds = load_dataset("FelicieGS/STHELAR_20x", split="train", streaming=True)
ds = ds.cast_column("image", Image(decode=True))

# Train: touch less of the stream; Val: do NOT shard
ds_tr = ds.shard(num_shards=64, index=0)
ds_va = ds

def filt_train(ex): return ex["Jaccard"] >= j_min and ex["tissue"] != heldout
def filt_val(ex):   return ex["Jaccard"] >= j_min and ex["tissue"] == heldout

def to_label(ex):
    counts = ex["cell_counts"]
    best = max(range(1, len(counts)), key=lambda i: counts[i])
    ex["y"] = labels[id2group.get(best, "Other")]
    return ex

def stream_take(base, match_fn, take, scan_cap):
    taken = scanned = 0
    for ex in base:
        scanned += 1
        if match_fn(ex):
            yield to_label(ex)
            taken += 1
            if taken >= take: break
        if scanned >= scan_cap: break

tfm = transforms.Compose([
    transforms.Resize(img_size, antialias=True),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class HFStream(IterableDataset):
    def __init__(self, make_iter, transform): self.make_iter, self.t = make_iter, transform
    def __iter__(self):
        for ex in self.make_iter():
            yield self.t(ex["image"].convert("RGB")), torch.tensor(ex["y"], dtype=torch.long)

# num_workers=1 avoids the HF “num_shards=1” warning cleanly
def make_loader(base, match_fn, n, scan_cap, bs):
    gen = lambda: stream_take(base, match_fn, n, scan_cap)
    return DataLoader(HFStream(gen, tfm), batch_size=bs, num_workers=0,
                      pin_memory=(device=="cuda"), persistent_workers=False)

train_loader = make_loader(ds_tr, filt_train, train_n, max_scan_train, batch_size)
val_loader   = make_loader(ds_va, filt_val,   val_n,   max_scan_val,   batch_size)

m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
m.fc = torch.nn.Linear(m.fc.in_features, len(GROUPS))
m = m.to(device).to(memory_format=torch.channels_last)

opt = torch.optim.AdamW(m.parameters(), lr=lr)
crit = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def run_epoch(loader, train=True):
    m.train(mode=train)
    total = correct = 0
    loss_sum = 0.0
    for x,y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        if train: opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device=="cuda")):
            out = m(x); loss = crit(out, y)
        if train:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += x.size(0)
    if total == 0:
        print("Warning: no samples yielded from loader.")
        return float("nan"), float("nan")
    return loss_sum/total, correct/total

for e in range(epochs):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    va_loss, va_acc = run_epoch(val_loader, False)
    print(f"epoch {e+1}/{epochs}  train {tr_loss:.4f}/{tr_acc:.3f}  val {va_loss:.4f}/{va_acc:.3f}")
