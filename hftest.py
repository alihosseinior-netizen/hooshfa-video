# Test free Hugging Face image-to-video Spaces. mode=api: print each Space's API. mode=run: animate test/horse.jpg and send to Telegram.
import os, sys, time, json, subprocess, traceback
from gradio_client import Client, handle_file
mode, spaces = sys.argv[1], sys.argv[2].split(',')
for sp in spaces:
    print('=====', sp, flush=True)
    try:
        c = Client(sp, token=os.environ['HF_TOKEN'], verbose=False)
        if mode == 'api': print(c.view_api(print_info=False, return_format='str')[:3000]); continue
        t = time.time()
        args = json.loads(os.environ['ARGS'])  # {"api_name": "...", "kwargs": {...}}; "IMG" value -> test image
        kw = {k: (handle_file('test/horse.jpg') if v == 'IMG' else v) for k, v in args['kwargs'].items()}
        r = c.predict(api_name=args['api_name'], **kw)
        print('done in', round(time.time() - t), 's ->', str(r)[:500])
        path = r[0] if isinstance(r, (list, tuple)) else r
        if isinstance(path, dict): path = path.get('video') or path.get('path') or path.get('value')
        subprocess.run(['curl', '-s', '-F', 'chat_id=' + os.environ['CHAT'], '-F', f'video=@{path}', '-F', f'caption=test {sp} {round(time.time()-t)}s',
                        f'https://api.telegram.org/bot{os.environ["BOT"]}/sendVideo'], check=False)
    except Exception as e: print('ERROR', repr(e)[:800]); traceback.print_exc(limit=2)
