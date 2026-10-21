#!/bin/bash
# hooshfa video renderer: 4 AI images -> Ken Burns zoom/pan clips -> crossfades -> soft music pad -> watermark -> send to Telegram.
# Inputs (env): EN (English prompt), CHAT (Telegram chat id), CAP (caption HTML), JOB, WORKER, VS (video secret), BOT (bot token).
set -euo pipefail
command -v ffmpeg >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg >/dev/null; }
mkdir -p f
SHOTS=("wide establishing shot" "medium shot" "close-up detail shot" "dramatic low angle shot")
for i in 0 1 2 3; do
  P="$EN, ${SHOTS[$i]}, cinematic lighting, highly detailed"
  ok=0
  for try in 1 2; do
    if curl -sf -G "$WORKER/vimg" -H "X-Video: $VS" --data-urlencode "p=$P" --data-urlencode "s=$((RANDOM * (i + 7)))" -o "f/$i.jpg" --max-time 120; then ok=1; break; fi
    sleep 5
  done
  [ $ok = 1 ] || { echo "image $i failed"; exit 1; }
done
# Each clip 4s at 25fps; alternate zoom in / zoom out / pan so it feels alive.
Z=("min(zoom+0.0020,1.35)" "if(eq(on,0),1.35,max(zoom-0.0020,1.0))" "1.25" "min(zoom+0.0015,1.30)")
X=("iw/2-(iw/zoom/2)" "iw/2-(iw/zoom/2)" "(iw-iw/zoom)*on/100" "iw/2-(iw/zoom/2)")
for i in 0 1 2 3; do
  ffmpeg -y -loglevel error -loop 1 -i "f/$i.jpg" -vf "scale=2048:2048,zoompan=z='${Z[$i]}':x='${X[$i]}':y='ih/2-(ih/zoom/2)':d=100:s=1024x1024:fps=25,format=yuv420p" -t 4 -c:v libx264 -preset veryfast "f/c$i.mp4"
done
# Crossfade chain (0.6s each): total 4*4 - 3*0.6 = 14.2s
ffmpeg -y -loglevel error -i f/c0.mp4 -i f/c1.mp4 -i f/c2.mp4 -i f/c3.mp4 -filter_complex \
 "[0][1]xfade=transition=fade:duration=0.6:offset=3.4[a];[a][2]xfade=transition=smoothleft:duration=0.6:offset=6.8[b];[b][3]xfade=transition=fade:duration=0.6:offset=10.2[v]" \
 -map "[v]" -c:v libx264 -preset veryfast f/v.mp4
# Soft ambient pad (A minor chord with slow swell) - our own sound, no copyright.
ffmpeg -y -loglevel error -f lavfi -i "aevalsrc='0.05*(sin(2*PI*220*t)+0.8*sin(2*PI*261.63*t)+0.7*sin(2*PI*329.63*t)+0.4*sin(2*PI*110*t))*(0.75+0.25*sin(2*PI*0.25*t))':s=44100:d=14.2" \
 -af "lowpass=f=1200,afade=t=in:d=1.5,afade=t=out:st=12.4:d=1.8,aecho=0.8:0.7:60:0.3" -c:a aac -b:a 128k f/a.m4a
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
WM="drawtext=fontfile=$FONT:text='@hooshfa_bot':fontcolor=white@0.85:fontsize=34:x=w-tw-28:y=h-th-28:shadowcolor=black@0.6:shadowx=2:shadowy=2"
[ -f "$FONT" ] || WM="null"
ffmpeg -y -loglevel error -i f/v.mp4 -i f/a.m4a -vf "$WM" -c:v libx264 -preset veryfast -crf 23 -c:a copy -shortest -movflags +faststart out.mp4
ls -la out.mp4
KB='{"inline_keyboard":[[{"text":"🔄 یه ویدیو دیگه","callback_data":"vagain"}]]}'
R=$(curl -s -F chat_id="$CHAT" -F video=@out.mp4 -F caption="$CAP" -F parse_mode=HTML -F supports_streaming=true -F width=1024 -F height=1024 -F duration=14 -F reply_markup="$KB" "https://api.telegram.org/bot$BOT/sendVideo")
echo "$R" | grep -q '"ok":true' || { echo "telegram send failed: $(echo "$R" | head -c 200)"; exit 1; }
