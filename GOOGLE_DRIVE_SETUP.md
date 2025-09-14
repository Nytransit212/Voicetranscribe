# Google Drive Integration Setup Guide

This guide explains how to set up Google Drive integration for large file uploads in the Ensemble Transcription System.

## Overview

The system uses Google Service Account authentication to upload large video files (up to 5TB) directly to your Google Drive, bypassing nginx proxy limitations that restrict uploads to 200MB.

## Features

- ✅ Upload files up to 5TB (bypasses nginx 413 errors)
- ✅ Resumable uploads for reliability with large files
- ✅ Service account authentication (no user OAuth required)
- ✅ Direct integration with existing transcription pipeline
- ✅ Progress tracking for uploads and downloads
- ✅ Support for multiple video formats (MP4, AVI, MOV, MKV, WebM)

## Step 1: Create Google Cloud Project and Service Account

### 1.1 Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note your project ID

### 1.2 Enable Google Drive API

1. In the Google Cloud Console, go to **APIs & Services > Library**
2. Search for "Google Drive API"
3. Click on it and press **Enable**

### 1.3 Create Service Account

1. Go to **IAM & Admin > Service Accounts**
2. Click **Create Service Account**
3. Enter details:
   - **Name**: `ensemble-transcription-drive`
   - **Description**: `Service account for ensemble transcription file uploads`
4. Click **Create and Continue**
5. Skip role assignment (not needed for Drive API)
6. Click **Done**

### 1.4 Create Service Account Key

1. Click on the created service account
2. Go to **Keys** tab
3. Click **Add Key > Create new key**
4. Select **JSON** format
5. Click **Create**
6. Save the downloaded JSON file securely
7. **Important**: Keep this file secure and never commit it to version control

## Step 2: Configure Google Drive Folder

### 2.1 Create Target Folder

1. Go to [Google Drive](https://drive.google.com)
2. Create a new folder for uploaded videos (e.g., "Ensemble Transcription Uploads")
3. Copy the folder ID from the URL:
   - URL: `https://drive.google.com/drive/folders/1ABC123xyz_YOUR_FOLDER_ID`
   - Folder ID: `1ABC123xyz_YOUR_FOLDER_ID`

### 2.2 Share Folder with Service Account

1. Right-click the folder and select **Share**
2. Add the service account email address (found in the JSON key file as `client_email`)
3. Give it **Editor** permissions
4. Click **Send**

**Important**: The service account email looks like:
`ensemble-transcription-drive@your-project-id.iam.gserviceaccount.com`

## Step 3: Environment Configuration

### 3.1 Prepare Service Account JSON

Convert the JSON key file to a single-line string:

**Method 1: Using jq (recommended)**
```bash
cat your-service-account-key.json | jq -c .
```

**Method 2: Manual formatting**
Remove all newlines and extra spaces from the JSON file to make it a single line.

### 3.2 Set Environment Variables

Create a `.env` file in your project root or set environment variables:

```bash
# Google Service Account JSON (entire JSON as single line)
GOOGLE_SERVICE_ACCOUNT_KEY='{"type":"service_account","project_id":"your-project","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n","client_email":"your-service-account@your-project.iam.gserviceaccount.com","client_id":"...","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"..."}'

# Google Drive Folder ID
GOOGLE_DRIVE_FOLDER_ID=1ABC123xyz_YOUR_FOLDER_ID
```

**Important Notes:**
- Replace `your-project` with your actual Google Cloud project ID
- Replace `1ABC123xyz_YOUR_FOLDER_ID` with your actual folder ID
- Ensure the JSON is properly escaped for environment variable storage
- In JSON, make sure to escape newlines in the private key as `\\n`

### 3.3 Replit Configuration

If using Replit:

1. Go to your Repl settings
2. Navigate to **Secrets** (environment variables)
3. Add the two environment variables:
   - Key: `GOOGLE_SERVICE_ACCOUNT_KEY`, Value: [your single-line JSON]
   - Key: `GOOGLE_DRIVE_FOLDER_ID`, Value: [your folder ID]

## Step 4: Verify Setup

### 4.1 Test Configuration

Run the application and check the Google Drive upload section:

1. The app will automatically verify your Google Drive setup
2. Look for the "Google Drive Configuration Status" section
3. You should see:
   - ✅ Google Drive API authenticated successfully
   - ✅ Service Account email displayed
   - ✅ Folder Access verified

### 4.2 Troubleshooting

**"Google Drive not configured" error:**
- Check that both environment variables are set
- Verify the JSON format is correct (single line, properly escaped)

**"Folder not found or no access" error:**
- Verify the folder ID is correct
- Ensure the folder is shared with the service account email
- Check that the service account has Editor permissions

**Authentication errors:**
- Verify the Google Drive API is enabled in Google Cloud Console
- Check that the service account JSON is valid
- Ensure the private key is properly escaped in the environment variable

## Step 5: Usage

### 5.1 Upload Files

1. Navigate to the main application page
2. Use the "Upload Video File to Google Drive" section
3. Choose between:
   - **Upload Local File**: Upload files up to 5GB from your computer
   - **Use Existing Google Drive File**: Use a file already in your Google Drive

### 5.2 File Processing

1. After upload/selection, configure processing parameters
2. Click "🚀 Start Ensemble Processing"
3. The system will:
   - Download the file from Google Drive (if needed)
   - Process through the ensemble transcription pipeline
   - Display results

### 5.3 Large File Handling

- Files over 10MB use resumable upload for reliability
- Upload progress is displayed with speed and ETA estimates
- Failed uploads can be resumed automatically
- Maximum file size: 5TB (limited by Google Drive)

## Security Considerations

### 5.1 Service Account Security

- ✅ Service account has minimal required permissions
- ✅ No user OAuth required - backend-only authentication
- ✅ JSON key stored as environment variable (not in code)
- ✅ Folder access limited to specific shared folder

### 5.2 Best Practices

1. **Never commit service account keys to version control**
2. **Regularly rotate service account keys**
3. **Use specific folder sharing (not entire Drive access)**
4. **Monitor service account usage in Google Cloud Console**
5. **Set up folder access notifications if needed**

## API Limits and Quotas

Google Drive API has the following limits:
- **Queries per 100 seconds per user**: 1,000
- **Queries per 100 seconds**: 100,000,000
- **Upload file size limit**: 5TB per file

These limits are sufficient for normal usage of the transcription system.

## Support

If you encounter issues:

1. Check the application logs for detailed error messages
2. Verify all setup steps are completed correctly
3. Test the configuration using the built-in verification
4. Ensure your Google Cloud project has billing enabled (required for API usage)

## Cost Considerations

- Google Drive API usage is typically free for normal usage
- Google Drive storage costs apply for uploaded files
- Consider setting up automatic file cleanup if storage costs are a concern