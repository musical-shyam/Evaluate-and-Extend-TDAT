import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# -------------------- config --------------------
attack_type = 'FGSM'          # set to 'PGD' for PGD‑10
batch_size  = 128
epsilon     = 8/255
alpha, pgd_iters = (2/255, 10) if attack_type == 'PGD' else (None, None)
ckpt_dir    = 'TDAT-Checkpoints'
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ------------------------------------------------

# ---------- data ----------
transform = transforms.ToTensor()          # models were trained with 32×32 inputs
testset   = datasets.CIFAR100(root='data', train=False,
                              download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# ---------- ResNet‑18 ----------
resnet = models.resnet18(weights=None)
resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet.maxpool = nn.Identity()
resnet.fc = nn.Linear(resnet.fc.in_features, 100)

def load_ckpt(model, path):
    """Load checkpoint, handle CPU/GPU and DataParallel prefixes."""
    ckpt = torch.load(path, map_location=device)
    # Accept both plain dict and dict with 'state_dict' / 'model_state_dict'
    state = ckpt
    for key in ('state_dict', 'model_state_dict'):
        if key in ckpt:
            state = ckpt[key]
            break
    # Strip 'module.' if it exists
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    return model

resnet_ckpt = os.path.join(ckpt_dir,
                'ResNet18_CIFAR100_TDAT_robustAcc_0.2226_clean_acc_0.6373.pt')
resnet = load_ckpt(resnet, resnet_ckpt).to(device).eval()

# ---------- DeiT‑Small ----------
try:
    import timm
except ImportError as e:
    raise ImportError("pip install timm") from e

# 32×32 variant; if you actually trained 224×224 just resize the inputs instead
deit = timm.create_model('deit_small_patch16_32',
                         pretrained=False, num_classes=100)
deit_ckpt = os.path.join(ckpt_dir,
              'DeiT-Small_CIFAR100_epoch_74_robustAcc_0.2128_cleanAcc_0.3674.pt')
deit = load_ckpt(deit, deit_ckpt).to(device).eval()

# ---------- attacks ----------
def fgsm_attack(model, images, labels, eps):
    images = images.clone().detach().to(device).requires_grad_(True)
    loss   = F.cross_entropy(model(images), labels.to(device))
    loss.backward()
    pert   = images + eps * images.grad.sign()
    return torch.clamp(pert.detach(), 0., 1.)

def pgd_attack(model, images, labels, eps, a, iters):
    ori   = images.clone().detach().to(device)
    pert  = ori.clone()
    for _ in range(iters):
        pert.requires_grad_(True)
        loss = F.cross_entropy(model(pert), labels.to(device))
        loss.backward()
        adv  = pert + a * pert.grad.sign()
        eta  = torch.clamp(adv - ori, -eps, eps)
        pert = torch.clamp(ori + eta, 0., 1.).detach()
    return pert

def make_adv(model, images, labels):
    if attack_type == 'FGSM':
        return fgsm_attack(model, images, labels, epsilon)
    else:
        return pgd_attack(model, images, labels, epsilon, alpha, pgd_iters)

# ---------- evaluation ----------
def correct(model, x, y):
    return (model(x.to(device)).argmax(1) == y.to(device)).sum().item()

with open('transferability.log', 'w') as log:
    log.write(f'Attack: {attack_type}\n')

    # -------- source = ResNet --------
    tot, rr, dr = 0, 0, 0
    for imgs, lbls in test_loader:
        adv = make_adv(resnet, imgs, lbls)
        rr += correct(resnet, adv, lbls)
        dr += correct(deit,  adv, lbls)
        tot += lbls.size(0)

    with open('transferability.log', 'a') as log:
        log.write(f'  ResNet on ResNet adversarial examples: {rr*100/tot:.2f}%\n')
        log.write(f'  DeiT  on ResNet adversarial examples: {dr*100/tot:.2f}%\n')

    # -------- source = DeiT --------
    tot, dd, rd = 0, 0, 0
    for imgs, lbls in test_loader:
        adv = make_adv(deit, imgs, lbls)
        dd += correct(deit,   adv, lbls)
        rd += correct(resnet, adv, lbls)
        tot += lbls.size(0)

    with open('transferability.log', 'a') as log:
        log.write(f'  DeiT  on DeiT  adversarial examples: {dd*100/tot:.2f}%\n')
        log.write(f'  ResNet on DeiT  adversarial examples: {rd*100/tot:.2f}%\n')

print('Evaluation finished – see transferability.log')
