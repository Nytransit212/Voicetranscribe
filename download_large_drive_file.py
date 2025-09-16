#!/usr/bin/env python3
"""
Enhanced Google Drive downloader for large files with virus scan bypass
"""
import requests
import re
import sys
import os

def download_large_drive_file(file_id, output_path):
    """Download large file from Google Drive, handling virus scan confirmation"""
    
    print(f'📥 Downloading file: {file_id}')
    
    # Initial download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    
    print('🔗 Making initial request...')
    response = session.get(url, stream=True)
    
    # Check if this is the virus scan warning page
    if response.status_code == 200:
        content_snippet = response.content[:2000].decode('utf-8', errors='ignore')
        
        if 'virus scan warning' in content_snippet.lower() or 'download anyway' in content_snippet.lower():
            print('⚠️  Large file detected - bypassing virus scan warning...')
            
            # Extract the confirmation token from the HTML
            token_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', content_snippet)
            if token_match:
                confirm_token = token_match.group(1)
                print(f'🔑 Found confirmation token: {confirm_token[:10]}...')
                
                # Use the confirmation token for direct download
                download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                print(f'📦 Downloading with confirmation token...')
                
                response = session.get(download_url, stream=True)
            else:
                # Try alternative method - look for direct download link
                link_match = re.search(r'href="(/uc\?export=download[^"]+)"', content_snippet)
                if link_match:
                    download_path = link_match.group(1)
                    download_url = f"https://drive.google.com{download_path}"
                    print(f'🔗 Found direct download link, downloading...')
                    response = session.get(download_url, stream=True)
                else:
                    print('❌ Could not find download token or link')
                    return None
    
    # Check final response
    if response.status_code != 200:
        print(f'❌ Download failed with status: {response.status_code}')
        return None
    
    # Check content type
    content_type = response.headers.get('content-type', '')
    if 'text/html' in content_type:
        print('❌ Still getting HTML instead of file - download failed')
        return None
    
    # Download the file
    total_size = int(response.headers.get('content-length', 0))
    print(f'📊 File size: {total_size / 1024 / 1024:.1f} MB')
    
    with open(output_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                # Progress update every 10MB
                if downloaded % (1024 * 1024 * 10) == 0:
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f'  📦 {progress:.1f}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)')
                    else:
                        print(f'  📦 Downloaded {downloaded/1024/1024:.1f} MB')
    
    final_size = os.path.getsize(output_path)
    print(f'✅ Download complete: {final_size/1024/1024:.1f} MB')
    
    return output_path

if __name__ == '__main__':
    file_id = '1VLXvaXYa0LMokEHPp-58ERX1wFcysWAu'
    output_path = '/tmp/cincinnati_swing.mp4'
    
    result = download_large_drive_file(file_id, output_path)
    if result:
        print(f'✅ File ready: {result}')
    else:
        print('❌ Download failed')