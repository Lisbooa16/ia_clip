from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

flow = InstalledAppFlow.from_client_secrets_file(
    "client_secret.JSON",  # JSON baixado do Google Cloud
    SCOPES,
)

creds = flow.run_local_server(port=0)

print("\n==============================")
print("COPIE ISSO PARA O settings.py")
print("==============================\n")

print('SOCIAL_PUBLISHING = {')
print('    "youtube": {')
print(f'        "client_id": "{creds.client_id}",')
print(f'        "client_secret": "{creds.client_secret}",')
print(f'        "refresh_token": "{creds.refresh_token}",')
print('        "token_uri": "https://oauth2.googleapis.com/token",')
print('        "scopes": ["https://www.googleapis.com/auth/youtube.upload"],')
print('    },')
print('}')
