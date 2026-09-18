"""Verify the active CLI OAuth identity without printing credentials or changing accounts."""
import argparse
from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.kaggle_creds import KaggleCredentials
from pathlib import Path

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('expected_user'); args=ap.parse_args()
    with KaggleClient() as client:
        creds=KaggleCredentials.load(client, file_path=str(Path.home()/'.kaggle/credentials.json'))
        creds._access_token=creds.generate_access_token().token
        user=creds.introspect()
        if user != args.expected_user:
            raise SystemExit(f'Expected {args.expected_user}, server authenticated {user}')
        print(f'Server-confirmed CLI OAuth account: {user}')
