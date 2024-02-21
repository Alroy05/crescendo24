# import webbrowser
# import gspread
# from oauth2client.service_account import ServiceAccountCredentials

# # Function to open the URL in a web browser
# def open_url_in_browser(url):
#     webbrowser.open(url)

# # Function to update Google Sheet with URL and run the command
# def update_google_sheet(url, command):
#     # Add your Google Sheets credentials JSON file path
#     credentials_path = 'credentials.json'
    
#     # Set up the credentials and access Google Sheets
#     scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
#     credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
#     gc = gspread.authorize(credentials)

#     # Open the Google Sheet using its URL
#     sheet_url = 'https://docs.google.com/spreadsheets/d/19knfUg6lbwL9ugfGoD-5oyRcObiLbAlKfn_SJi5KtHI'
#     sheet = gc.open_by_url(sheet_url)

#     # Select the desired worksheet (assuming it's the first sheet)
#     worksheet = sheet.get_worksheet(0)

#     # Update A2 cell with the provided URL
#     worksheet.update_acell('A2', url)

#     # Update B2 cell with the provided command
#     worksheet.update_acell('B2', command)

# # Get URL input from the user
# url = input("Enter the URL: ")

# # Run the function to open the URL in a web browser
# open_url_in_browser(url)

# # Get command input from the user
# command = "=IMPORTFROMWEB(A2,B1:D1)"

# # Run the function to update the Google Sheet
# update_google_sheet(url, command)

# print("Google Sheet updated successfully.")





import webbrowser
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Function to open the URL in a web browser
def open_url_in_browser(url):
    webbrowser.open(url)

# Function to update Google Sheet with URL and run the command
def update_google_sheet(url, command):
    # Add your Google Sheets credentials JSON file path
    credentials_path = 'credentials.json'
    
    # Set up the credentials and access Google Sheets
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    gc = gspread.authorize(credentials)

    # Open the Google Sheet using its URL
    sheet_url = 'https://docs.google.com/spreadsheets/d/19knfUg6lbwL9ugfGoD-5oyRcObiLbAlKfn_SJi5KtHI'
    sheet = gc.open_by_url(sheet_url)

    # Select the desired worksheet (assuming it's the first sheet)
    worksheet = sheet.get_worksheet(0)

    # Update A2 cell with the provided URL
    worksheet.update_acell('A2', url)

    # Update B2 cell with the provided command
    worksheet.update_acell('B2', command)

# Function to download the updated Google Sheet in CSV format
def download_google_sheet(sheet, file_name):
    # Download the content of the sheet
    content = sheet.get_all_records()

    # Save the content locally in a CSV file with utf-8 encoding
    with open(file_name, 'w', encoding='utf-8', newline='') as file:
        # Use csv.writer to handle encoding and newlines properly
        import csv
        csv_writer = csv.writer(file)
        
        # Write header
        header = content[0].keys()
        csv_writer.writerow(header)
        
        # Write data
        for row in content:
            csv_writer.writerow(row.values())

# Get URL input from the user
url = input("Enter the URL: ")

# Run the function to open the URL in a web browser
open_url_in_browser(url)

# Get command input from the user
command = "=IMPORTFROMWEB(A2,B1:D1)"

# Run the function to update the Google Sheet
update_google_sheet(url, command)

# Specify the file name for the downloaded sheet
downloaded_file_name = 'data_sheet.csv'

credentials_path = 'credentials.json'
    
    # Set up the credentials and access Google Sheets
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
gc = gspread.authorize(credentials)
# Open the Google Sheet using its URL
sheet_url = 'https://docs.google.com/spreadsheets/d/19knfUg6lbwL9ugfGoD-5oyRcObiLbAlKfn_SJi5KtHI'
sheet = gc.open_by_url(sheet_url)

# Select the desired worksheet (assuming it's the first sheet)
worksheet = sheet.get_worksheet(0)

# Run the function to download the Google Sheet in CSV format
download_google_sheet(worksheet, downloaded_file_name)

print(f"Google Sheet updated successfully. Downloaded as '{downloaded_file_name}'.")

