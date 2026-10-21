# hooshfa video v2: AI images -> depth map (Depth Anything V2, CPU) -> real 3D parallax camera moves -> mood particles
# (rain/snow/fireflies/fog/dust/light) -> crossfades -> colour grade + Persian title + watermark -> our own music -> Telegram.
# Env: PLAN (json: title, mood, music, scenes[]), CHAT, CAP, JOB, WM ("1"/"0"), WORKER, VS, BOT, VOICE (optional piper voice), NARR.
import json, os, subprocess, sys, math, random, concurrent.futures as cf, urllib.request, urllib.parse
import numpy as np, cv2

W, H, FPS, L, XF = 720, 1280, 30, 4.2, 0.6
SW, SH = 792, 1408  # source canvas (10% margin for camera moves)
plan = json.loads(os.environ['PLAN'])
scenes = plan['scenes'][:14]
L = 4.6 if len(scenes) > 6 else 4.2  # ~1 minute for 12-14 scenes
mood, mkind = plan.get('mood', 'none'), plan.get('music', 'calm')
os.makedirs('f', exist_ok=True)
rnd = random.Random(str(plan.get('seed') or os.environ.get('JOB', '1')))  # same seed = same images (paid no-watermark re-render)

def sh(cmd): subprocess.run(cmd, shell=True, check=True)

# ---------- 1. images (Worker chain: FLUX-2 vertical -> FLUX-1 -> Pollinations) ----------
def fetch(i):
    q = urllib.parse.urlencode({'p': scenes[i] + ', cinematic lighting, highly detailed, vertical composition', 's': rnd.randint(1, 10**9), 'w': 768, 'h': 1344})
    for _ in range(3):
        try:
            req = urllib.request.Request(os.environ['WORKER'] + '/vimg?' + q, headers={'X-Video': os.environ['VS'], 'User-Agent': 'hooshfa-video/2 (+github actions)'})
            resp = urllib.request.urlopen(req, timeout=150); data = resp.read(); print('img', i, resp.headers.get('x-by'), flush=True)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None: return img
        except Exception as e: print('img retry', i, e)
    raise SystemExit(f'image {i} failed')

def fit(img):  # cover-crop to SWxSH
    h, w = img.shape[:2]; s = max(SW / w, SH / h)
    img = cv2.resize(img, (math.ceil(w * s), math.ceil(h * s)), interpolation=cv2.INTER_LANCZOS4)
    y, x = (img.shape[0] - SH) // 2, (img.shape[1] - SW) // 2
    return img[y:y + SH, x:x + SW]

with cf.ThreadPoolExecutor(4) as ex: imgs = [fit(x) for x in ex.map(fetch, range(len(scenes)))]

# ---------- 3b. hero shots: REAL AI motion (Wan 2.2 image-to-video on a free Hugging Face ZeroGPU Space), best effort ----------
HERO_SPACE = 'zerogpu-aoti/wan2-2-fp8da-aoti-faster'
motions = plan.get('motions') or []
hero_idx = [i for i in (plan.get('hero') or [2, len(scenes) - 2]) if isinstance(i, int) and 0 <= i < len(scenes)][:2]
def hero(i):
    try:
        import shutil
        from gradio_client import Client, handle_file
        cv2.imwrite(f'f/s{i}.jpg', imgs[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
        c = Client(HERO_SPACE, token=os.environ.get('HF_TOKEN') or None, verbose=False)
        mo = motions[i] if i < len(motions) and motions[i] else 'natural cinematic motion, the subject moves, camera slowly follows'
        r = c.predict(input_image=handle_file(f'f/s{i}.jpg'), prompt=str(mo)[:300] + ', smooth, cinematic, high quality', steps=6, duration_seconds=5, api_name='/generate_video')
        p = r[0] if isinstance(r, (list, tuple)) else r
        if isinstance(p, dict): p = p.get('video') or p.get('path')
        shutil.copy(p, f'f/h{i}.mp4'); print('hero', i, 'ok', flush=True); return True
    except Exception as e: print('hero', i, 'skipped:', repr(e)[:300], flush=True); return False
hero_pool = cf.ThreadPoolExecutor(2); hero_jobs = {i: hero_pool.submit(hero, i) for i in hero_idx}

# ---------- 2. depth (bigger = nearer) ----------
import onnxruntime as ort
sess = ort.InferenceSession('models/depth.onnx', providers=['CPUExecutionProvider'])
def depth(img):
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (518, 518)).astype(np.float32) / 255
    x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
    d = sess.run(None, {sess.get_inputs()[0].name: x})[0].squeeze()
    d = cv2.resize(d, (SW, SH)); d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return cv2.GaussianBlur(d, (0, 0), 9).astype(np.float32)  # soft edges = less tearing

# ---------- 3. particles ----------
class Particles:
    def __init__(self, kind):
        self.k = kind; n = {'rain': 260, 'snow': 170, 'sparkle': 70, 'dust': 90}.get(kind, 0)
        self.p = np.column_stack([np.random.rand(n) * W, np.random.rand(n) * H, np.random.rand(n)]) if n else None
        if kind == 'fog':
            f = cv2.resize(np.random.rand(18, 10).astype(np.float32), (W * 2, H), interpolation=cv2.INTER_CUBIC)
            self.fog = cv2.GaussianBlur(f, (0, 0), 60)
    def draw(self, fr, t):
        k, o = self.k, None
        if k == 'rain':
            o = np.zeros_like(fr); p = self.p; p[:, 1] = (p[:, 1] + 38 + 20 * p[:, 2]) % H; p[:, 0] = (p[:, 0] + 6) % W
            for x, y, z in p: cv2.line(o, (int(x), int(y)), (int(x - 6), int(y - 26 - 14 * z)), (200, 200, 210), 1, cv2.LINE_AA)
            return cv2.addWeighted(fr, 1, o, 0.35, 0)
        if k == 'snow':
            o = np.zeros_like(fr); p = self.p; p[:, 1] = (p[:, 1] + 1.5 + 2.5 * p[:, 2]) % H; p[:, 0] = (p[:, 0] + np.sin(t * 2 + p[:, 2] * 9) * 0.8) % W
            for x, y, z in p: cv2.circle(o, (int(x), int(y)), int(1 + 3 * z), (255, 255, 255), -1, cv2.LINE_AA)
            return cv2.addWeighted(fr, 1, cv2.GaussianBlur(o, (0, 0), 1.2), 0.8, 0)
        if k in ('sparkle', 'dust'):
            o = np.zeros_like(fr); p = self.p; p[:, 1] = (p[:, 1] - 0.6 - p[:, 2]) % H; p[:, 0] = (p[:, 0] + np.sin(t + p[:, 2] * 7) * 0.7) % W
            col = (120, 230, 255) if k == 'sparkle' else (210, 225, 235)
            for x, y, z in p:
                a = 0.5 + 0.5 * math.sin(t * 5 + z * 20)
                cv2.circle(o, (int(x), int(y)), int(2 + 3 * z) if k == 'sparkle' else 1 + int(z * 2), tuple(int(c * a) for c in col), -1, cv2.LINE_AA)
            return cv2.add(fr, cv2.GaussianBlur(o, (0, 0), 3 if k == 'sparkle' else 1))
        if k == 'fog':
            sh_ = int(t * 25) % W; f = self.fog[:, sh_:sh_ + W]
            return (fr * (1 - 0.28 * f[..., None]) + 235 * 0.28 * f[..., None]).astype(np.uint8)
        if k == 'light':
            x = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]; c = 0.5 + 0.5 * math.sin(t * 0.8)
            g = np.exp(-((x - c) ** 2) / 0.05) * 60
            return np.clip(fr + g * np.array([0.4, 0.75, 1.0], np.float32), 0, 255).astype(np.uint8)
        return fr

# ---------- 4. parallax clips ----------
ease = lambda t: t * t * (3 - 2 * t)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
MOVES = ['push', 'pan', 'pull', 'rise']
def clip(i, img, d):
    move = MOVES[i % 4]; n = int(L * FPS)
    bx, by = (xx - W / 2) + SW / 2, (yy - H / 2) + SH / 2   # 1:1 base view (centre)
    dO = cv2.remap(d, bx, by, cv2.INTER_LINEAR)               # depth seen at each output pixel
    parts = Particles(mood)
    out = subprocess.Popen(f'ffmpeg -y -loglevel error -f rawvideo -pix_fmt bgr24 -s {W}x{H} -r {FPS} -i - -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p f/c{i}.mp4', shell=True, stdin=subprocess.PIPE)
    for k in range(n):
        t = ease(k / (n - 1)); dz = dO - 0.5
        if move == 'push': Z = 1 + t * (0.05 + 0.12 * dO); px, py = t * 10 * dz, 0
        elif move == 'pull': Z = 1 + (1 - t) * (0.05 + 0.12 * dO); px, py = 0, (1 - t) * 8 * dz
        elif move == 'pan': Z = 1.04 + 0.02 * dO; px, py = (t - 0.5) * 70 * (0.35 + dO), 0
        else: Z = 1.04 + 0.03 * t * dO; px, py = 0, (0.5 - t) * 60 * (0.35 + dO)
        mx = SW / 2 + (xx - W / 2) / Z - px
        my = SH / 2 + (yy - H / 2) / Z - py
        fr = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out.stdin.write(parts.draw(fr, k / FPS).tobytes())
    out.stdin.close(); out.wait()

for i, img in enumerate(imgs): clip(i, img, depth(img)); print('clip', i, 'ok', flush=True)
for i, j in hero_jobs.items():  # swap in real-motion clips that finished (parallax clip stays as fallback)
    try:
        if not j.result(timeout=420): continue
        cap = cv2.VideoCapture(f'f/h{i}.mp4'); dur = cap.get(7) / max(cap.get(5), 1); cap.release()
        k = max(1.0, L / max(dur, 0.5))
        sh(f"ffmpeg -y -loglevel error -i f/h{i}.mp4 -vf \"setpts={k:.3f}*PTS,scale=-2:1280:flags=lanczos,crop=720:1280,unsharp=5:5:0.8,fps={FPS},format=yuv420p\" -t {L} -an -c:v libx264 -preset veryfast -crf 18 f/c{i}.mp4")
        print('hero', i, 'used', flush=True)
    except Exception as e: print('hero', i, 'not used', repr(e)[:200])
hero_pool.shutdown(wait=False, cancel_futures=True)

TR = ['fade', 'smoothleft', 'circleopen', 'fadeblack']
n = len(imgs); T = n * L - (n - 1) * XF
chain, last = [], '[0]'
for i in range(1, n):
    lab = f'[x{i}]' if i < n - 1 else '[v]'
    chain.append(f'{last}[{i}]xfade=transition={TR[i % 4]}:duration={XF}:offset={i * (L - XF):.2f}{lab}'); last = lab
sh('ffmpeg -y -loglevel error ' + ' '.join(f'-i f/c{i}.mp4' for i in range(n)) + f' -filter_complex "{";".join(chain)}" -map "[v]" -c:v libx264 -preset veryfast -crf 18 f/v.mp4')

# ---------- 5. music (our own synthesis: no copyright claims on Instagram/YouTube) ----------
import music
music.make(kind=mkind, seconds=T, path='f/m.wav', seed=rnd.randint(0, 10**6))
audio = 'f/m.wav'
if os.environ.get('VOICE') and os.environ.get('NARR'):
    try:
        sh(f'bash voice.sh {os.environ["VOICE"]} "{os.environ["NARR"]}" f/n.wav')
        sh('ffmpeg -y -loglevel error -i f/m.wav -i f/n.wav -filter_complex "[1]adelay=700|700,volume=1.6[n];[0]volume=0.35[m];[m][n]amix=inputs=2:duration=first:normalize=0" f/mix.wav')
        audio = 'f/mix.wav'
    except Exception as e: print('voice skipped', e)

# ---------- 6. title + grade + watermark ----------
title = (plan.get('title') or '').replace('\\', '').replace('{', '').replace('}', '')[:40]
ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: T,Vazirmatn,64,&H00FFFFFF,&H64000000,&H96000000,1,0,1,3,4,8,40,40,250,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.40,0:00:03.40,T,,0,0,0,,{{\\fad(500,600)}}{title}
"""
open('f/t.ass', 'w', encoding='utf-8').write(ass)
vf = ['eq=contrast=1.06:saturation=1.12', 'vignette=PI/5', 'unsharp=5:5:0.6', 'noise=alls=3:allf=t']
if title: vf.append('subtitles=f/t.ass:fontsdir=fonts')
if os.environ.get('WM', '1') == '1':
    vf.append("drawtext=fontfile=fonts/Vazirmatn-Bold.ttf:text='@hooshfa_bot':fontcolor=white@0.85:fontsize=30:x=w-tw-26:y=h-th-60:shadowcolor=black@0.6:shadowx=2:shadowy=2")
sh(f'ffmpeg -y -loglevel error -i f/v.mp4 -i {audio} -vf "{",".join(vf)}" -c:v libx264 -preset medium -crf 21 -c:a aac -b:a 160k -shortest -movflags +faststart out.mp4')
print('out.mp4', os.path.getsize('out.mp4'))

# ---------- 7. send ----------
kb = {'inline_keyboard': [[{'text': '🔄 یه ویدیو دیگه', 'callback_data': 'vagain'}]]}
r = subprocess.run(['curl', '-s', '-F', 'chat_id=' + os.environ['CHAT'], '-F', 'video=@out.mp4', '-F', 'caption=' + os.environ['CAP'], '-F', 'parse_mode=HTML',
    '-F', 'supports_streaming=true', '-F', f'width={W}', '-F', f'height={H}', '-F', f'duration={int(T)}', '-F', 'reply_markup=' + json.dumps(kb),
    f'https://api.telegram.org/bot{os.environ["BOT"]}/sendVideo'], capture_output=True, text=True).stdout
if '"ok":true' not in r: raise SystemExit('telegram send failed: ' + r[:200])
try: print('FILE_ID', json.loads(r)['result']['video']['file_id'])
except Exception: pass
