# Use gdown to download file and unzip it
import gdown
import zipfile
import os
import shutil

def download_and_unzip(url, output_dir):
    gdown.download(url, 'temp.zip', quiet=False)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.system('unzip temp.zip -d ' + output_dir)
    os.remove('temp.zip')

def download_folder(file_id, output_dir):
    """Download folder from google drive"""
    os.system(f"gdown https://drive.google.com/drive/folders/{file_id} -O {output_dir} --folder")


def download_pretrained_model(output_dir):
    """Download all pretrained models"""
    file_id = '18tmH0stc1z_TzfAHbQDu5HASNkaWFKk_'  # (300M)
    download_folder(file_id, output_dir)

def download_init_target(env_name, output_dir):
    """Download init and target for all environments"""
    urls = {
        'LiftSpread-v1': 'https://drive.google.com/uc?id=1b4Qw6cbWbtEiP3MO7v6WOLkgCplFnDb1',
        'CutRearrange-v1': 'https://drive.google.com/uc?id=1XFGKwngAX_4gVzmP38DMv0mo44a9e3Sd',
        'CutRearrangeSpread-v1': 'https://drive.google.com/uc?id=1IEAev3VeCXAKVEeZmYlWjlT2MHLWyedl',
    }
    url = urls[env_name]
    download_and_unzip(url, output_dir)

def download_demonstration(env_name, output_dir):
    """Download init and target for all environments"""
    urls = {
        'LiftSpread-v1': 'https://drive.google.com/uc?id=1gRaaMEG6ytmjOOqG-K79kMteGhXW_QFB',  # (6G)
        'CutRearrange-v1': 'https://drive.google.com/uc?id=1wKgFLSkdueVGoy_vDGxPiOnVwENKsSZT',  # (5G)
        'CutRearrangeSpread-v1': 'https://drive.google.com/uc?id=1OFcC8po2ZBmgMxtJ5C7uiWPG7ESm5W2v',  # (3G)
    }
    url = urls[env_name]
    download_and_unzip(url, output_dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, choices=['init_target', 'demo', 'pretrained'])
    parser.add_argument(
        '--env_name', type=str, default='CutRearrangeSpread', help='The environment to download',
        choices=['LiftSpread-v1', 'CutRearrange-v1', 'CutRearrangeSpread-v1', 'all'])
    args = parser.parse_args()

    if args.env_name == 'all':
        envs = ['LiftSpread-v1', 'CutRearrange-v1', 'CutRearrangeSpread-v1']
    else:
        envs = [args.env_name]

    if args.command == 'init_target':
        download_init_target(args.env_name, output_dir='./datasets/')
    elif args.command == 'demo':
        download_demonstration(args.env_name, output_dir='./data/')
    elif args.command == 'pretrained':
        download_pretrained_model(output_dir='./data/')
