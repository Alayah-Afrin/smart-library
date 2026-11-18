# Put Smart Library online (free, permanent link)

The website runs free on **Hugging Face Spaces** ("CPU basic": 2 CPUs, 16 GB memory; the site needs about 3 GB).
You do these steps once (about 30 minutes, mostly waiting). The parts that need **your** account and password
are marked 🔑, because nobody else should type those.

Open a terminal (PowerShell) in `D:\documents\Smart_Library_V2` for every command below.

---

## 1. 🔑 Make a free Hugging Face account
Go to https://huggingface.co/join, sign up, and confirm your email. Remember your **username**
(below it is written as `YOURNAME`; replace it with yours everywhere).

## 2. 🔑 Make an access token
huggingface.co → your picture (top right) → **Settings** → **Access Tokens** → **Create new token**
→ type **Write** → name it `smart-library` → **Create**. Copy the token (it starts with `hf_`).
Treat it like a password: never paste it into a chat or a file.

## 3. 🔑 Log in on this computer
```
.\venv\Scripts\hf.exe auth login
```
Paste the token when it asks (nothing shows while you paste; that's normal), press Enter, then answer `n`
to "add token as git credential".

## 4. Upload the book data (about 700 MB, private)
```
.\venv\Scripts\python.exe scripts\upload_data.py YOURNAME/smart-library-data
```
This makes a **private** dataset only you can see. It takes 5–20 minutes, depending on your internet.

## 5. Make the Space (the website)
https://huggingface.co/new-space
- Space name: `smart-library`
- SDK: **Docker** → **Blank**
- Hardware: **CPU basic · FREE**
- Public
→ **Create Space**

## 6. 🔑 Tell the Space where the data is
On the Space page → **Settings** → **Variables and secrets**:
- **New variable**: name `DATA_REPO`, value `YOURNAME/smart-library-data`
- **New secret**: name `HF_TOKEN`, value: your token from step 2 (or a new **Read** token)

## 7. Send the code
```
git push https://huggingface.co/spaces/YOURNAME/smart-library publish:main
```
If a login window appears: username = `YOURNAME`, password = your token.

## 8. Wait for it to build
On the Space page you'll see **Building** (about 10–15 min the first time), then **Running**.
Your site is at **https://YOURNAME-smart-library.hf.space**: share that link.

---

## Good to know
- **Free tier sleep:** if nobody visits for 48 hours the Space sleeps; the next visitor wakes it (takes ~2 minutes).
- **Updating later:** after changing code, run step 7 again. After changing book data, run step 4 again,
  then on the Space page click **Settings → Factory rebuild**.
- **Reviews and shelves:** "My shelf" lives in each visitor's own browser. Reader reviews are stored on the
  Space and are reset when it rebuilds.
- **Admin page:** set a secret `ADMIN_PASSWORD` in step 6 to use `/admin` (moderating reviews).
- **If the build fails:** open the **Logs** tab on the Space page and copy the last red lines.
