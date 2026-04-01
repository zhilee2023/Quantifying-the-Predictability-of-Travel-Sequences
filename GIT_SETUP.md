# Git 连接与推送（你来登录 / 配密钥）

仓库根目录假设为：

`C:\Users\zhile\Desktop\rl_c\Quantifying-the-Predictability-of-Travel-Sequences`

---

## 1. 进入目录

```powershell
cd "C:\Users\zhile\Desktop\rl_c\Quantifying-the-Predictability-of-Travel-Sequences"
```

---

## 2. 绑定远端（二选一）

### 方式 A：HTTPS（用 Personal Access Token 当密码）

```powershell
git remote remove origin 2>$null
git remote add origin https://github.com/zhilee2023/Quantifying-the-Predictability-of-Travel-Sequences.git
git remote -v
```

推送时若弹出登录：**用户名**填 `zhilee2023`，**密码**填你在 GitHub 生成的 **PAT**（不是账号密码）。

若 Windows 记住了错误账号（例如 `zspeech`），先清再推：

```powershell
cmdkey /list | findstr git
# 若看到 github.com 条目：
cmdkey /delete:LegacyGeneric:target=git:https://github.com
```

或用 **Git Credential Manager** 在下次 `git push` 时选 **Sign in with browser** 并登录 `zhilee2023`。

### 方式 B：SSH（推荐，免每次输 PAT）

1. 本机生成密钥（若已有可跳过）：

```powershell
ssh-keygen -t ed25519 -C "your_email@example.com" -f "$env:USERPROFILE\.ssh\id_ed25519_github" -N ""
```

2. 把 `~/.ssh/id_ed25519_github.pub` 全文复制到 GitHub → **Settings → SSH and GPG keys → New SSH key**（确保该 key 加在 **`zhilee2023`** 账号下）。

3. 配置 SSH 使用这把 key（可选，写入 `~/.ssh/config`）：

```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
```

4. 改远端并测：

```powershell
git remote set-url origin git@github.com:zhilee2023/Quantifying-the-Predictability-of-Travel-Sequences.git
ssh -T git@github.com
```

---

## 3. LFS（本仓库有大文件）

```powershell
git lfs install
git lfs status
```

---

## 4. 提交（若还有未提交的改动）

```powershell
git status
git add -A
git commit -m "Your message"
```

若本地 **已经 commit 过**，可跳过本节，直接推送。

---

## 5. 推送到 GitHub

```powershell
git branch -M main
git push -u origin main
```

首次推送若含 **Git LFS** 对象，时间会较长；若提示 **LFS 配额不足**，需在 GitHub 仓库 **Settings → Billing → Git LFS** 购买 Data pack，或改为只推代码、大文件另存 Release。

---

## 6. 一键脚本（可选）

在资源管理器中右键 **用 PowerShell 打开** 本仓库根目录，执行：

```powershell
.\git_push.ps1
```

（仅在你已配好 `origin` 且能 `ssh -T` 或已登录 HTTPS 时使用。）
