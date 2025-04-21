# Deepfake Detection Demo Client

## Overview
This repository contains the client application for interacting with the deepfake detection service. It captures video frames from Zoom calls or other video sources and sends them to the deepfake detection server for analysis.

## Features
- Screen capture functionality for Zoom and other video conferencing tools
- Secure communication with the confidential deepfake detection server

## Prerequisites
- Python 3.8+
- PIL (Python Imaging Library)
- Tinfoil client library (for confidential deployment)

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/tinfoilsh/df-demo-client.git
   cd df-demo-client
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

```python
python3 client.py
```

## How It Works
The client application:
1. Captures frames from your screen during video calls
2. Processes and encodes the frames
3. Sends them to the deepfake detection server
4. Receives and displays the analysis results

## Code Example
Using Tinfoil's secure client:

```python
# Without client verification
response = requests.post(url, headers=headers, data=payload, timeout=30)

# With client verification
import tinfoil
tfclient = tinfoil.NewSecureClient("enclave url", "repo")
response = tfclient.post(url, headers=headers, data=payload, timeout=30)
```

## Privacy Considerations
When using the confidential mode with Tinfoil:
- All video data remains encrypted end-to-end
- The analysis happens in a secure enclave
- Not even the service provider, or Tinfoil, can access your sensitive video content

## Related Repositories
- [confidential-df-demo](https://github.com/tinfoilsh/confidential-df-demo): Confidential deployment of the deepfake detection server
- [df-demo](https://github.com/tinfoilsh/df-demo): Core deepfake detection server implementation

## Contact
For more information about Tinfoil's confidential computing platform, visit [tinfoil.sh](https://tinfoil.sh)
