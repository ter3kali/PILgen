import os
EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
n = len([f for f in os.listdir('raw_images') if f.lower().endswith(EXTS)])
os.makedirs('data_init/programs', exist_ok=True)
for i in range(1, n + 1):
    open(f'data_init/programs/{i:04d}.ipg', 'w').write('size 64\nbg 0 0 0\n')
