from __future__ import print_function
from googleapiclient.discovery import build
from google.oauth2 import service_account

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

try:
    credentials = service_account.Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
except ImportError:
    credentials = service_account.Credentials.from_service_account_file('sheets/credentials.json', scopes=SCOPES)
    pass
except FileNotFoundError:
    credentials = service_account.Credentials.from_service_account_file('split_experiment/sheets/credentials.json', scopes=SCOPES)
    pass

spreadsheet_service = build('sheets', 'v4', credentials=credentials)
drive_service = build('drive', 'v3', credentials=credentials)
