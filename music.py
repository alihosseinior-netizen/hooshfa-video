# Our own background music, synthesised in numpy per mood (pad + arpeggio + bass + drums + reverb). No copyright claims.
import numpy as np, wave

SR = 44100
MOODS = {  # chords as (root midi, minor?) ; bpm ; drums ; arp speed (notes per beat)
    'epic':     ([(57, 1), (53, 0), (60, 0), (55, 0)], 88, 'epic', 2),
    'calm':     ([(62, 0), (57, 0), (59, 1), (55, 0)], 68, None, 1),
    'happy':    ([(60, 0), (55, 0), (57, 1), (53, 0)], 112, 'pop', 2),
    'mystery':  ([(52, 1), (57, 1), (60, 0), (59, 0)], 74, None, 1),
    'romantic': ([(53, 0), (57, 1), (58, 0), (60, 0)], 72, None, 1),
}
f = lambda m: 440.0 * 2 ** ((m - 69) / 12)

def tone(freq, n, harm=6, bright=1.0, detune=0.0):
    t = np.arange(n) / SR; out = np.zeros(n)
    for k in range(1, harm + 1):
        for d in ((-detune, detune) if detune else (0,)):
            out += np.sin(2 * np.pi * freq * k * (1 + d) * t) * (bright ** (k - 1)) / k
    return out

def env(n, a, r):
    e = np.ones(n); a, r = int(a * SR), int(r * SR)
    if a: e[:a] = np.linspace(0, 1, a)
    if r: e[-r:] *= np.linspace(1, 0, r)
    return e

def reverb(x, seconds=2.2, wet=0.32, seed=0):
    rng = np.random.default_rng(seed); n = int(seconds * SR)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (SR * seconds / 5)); ir /= np.sqrt((ir ** 2).sum())
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[:len(x)]
    return x * (1 - wet) + y * wet * 3

def make(kind='calm', seconds=15.0, path='m.wav', seed=0):
    chords, bpm, drums, arps = MOODS.get(kind, MOODS['calm'])
    rng = np.random.default_rng(seed)
    N = int(seconds * SR); beat = 60 / bpm; bar = 4 * beat
    pad, arp, bass, drum = (np.zeros(N) for _ in range(4))
    nb = int(np.ceil(seconds / bar))
    for b in range(nb):
        root, minor = chords[b % len(chords)]
        notes = [root, root + (3 if minor else 4), root + 7]
        s = int(b * bar * SR); n = min(int(bar * SR), N - s)
        if n <= 0: break
        for m in notes + [root + 12]:
            pad[s:s + n] += tone(f(m - 12), n, harm=7, bright=0.62, detune=0.003) * env(n, 0.6, 0.5) * 0.10
        bass[s:s + n] += tone(f(root - 24), n, harm=3, bright=0.4) * env(n, 0.02, 0.3) * 0.32
        step = beat / arps; seq = notes + [root + 12, notes[1] + 12] if arps == 2 else notes + [root + 12]
        for j in range(int(bar / step)):
            ss = s + int(j * step * SR); nn = min(int(step * 1.8 * SR), N - ss)
            if nn <= 0: continue
            m = seq[j % len(seq)] + 12
            arp[ss:ss + nn] += tone(f(m), nn, harm=3, bright=0.35) * np.exp(-np.arange(nn) / SR * (7 if arps == 2 else 4)) * 0.11
        if drums:
            for q in range(4):
                ss = s + int(q * beat * SR); nn = min(int(0.45 * SR), N - ss)
                if nn <= 0: continue
                tt = np.arange(nn) / SR
                kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
                if drums == 'pop' or q in (0, 2): drum[ss:ss + nn] += kick * (0.55 if drums == 'pop' else 0.75)
                if drums == 'pop':  # off-beat hats
                    hs = ss + int(beat / 2 * SR); hn = min(int(0.06 * SR), N - hs)
                    if hn > 0: drum[hs:hs + hn] += np.diff(rng.standard_normal(hn + 1)) * np.exp(-np.arange(hn) / SR * 60) * 0.05
                if drums == 'epic' and q == 3 and b % 2 == 1:
                    tom = np.sin(2 * np.pi * (90 + 60 * np.exp(-tt * 12)) * tt) * np.exp(-tt * 5) * 0.5
                    drum[ss:ss + nn] += tom
    wet = 0.45 if kind in ('mystery', 'calm', 'romantic') else 0.3
    left = reverb(pad + arp * 0.9, wet=wet, seed=seed) + bass + drum
    right = reverb(np.roll(pad, int(0.012 * SR)) + arp * 1.1, wet=wet, seed=seed + 1) + bass + drum
    st = np.stack([left, right], 1)
    fade = np.ones(N); fade[:SR] = np.linspace(0, 1, SR); fade[-2 * SR:] = np.linspace(1, 0, 2 * SR)
    st *= fade[:, None]; st = np.tanh(st * 1.2); st /= np.abs(st).max() + 1e-9; st *= 0.89
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st * 32767).astype('<i2').tobytes())

if __name__ == '__main__':
    import sys; make(sys.argv[1] if len(sys.argv) > 1 else 'calm', 15, sys.argv[2] if len(sys.argv) > 2 else 'm.wav')
