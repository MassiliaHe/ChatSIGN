
# Sign Language Communication with ChatGPT

This repository contains the code for the Sign Language Communication Application "ChatSign". This application bridges the gap between sign language users and the digital world, providing a seamless platform for communication in American Sign Language (ASL).

The application enables users to express their requests using the ASL alphabet. These requests are then sent to ChatGPT, which processes and responds. The response is subsequently retranscribed into sign language animations, making the communication process intuitive and accessible for ASL users.

<img src="/images/chatsign.jpg" width="600" height="500">

# Overview

This repository is structured into three main components:

- **Sign2Text**: This module focuses on translating sign language gestures into text. It uses gesture recognition technology to accurately interpret the ASL alphabet.

- **Communication with ChatGPT**: This module sends the translated text requests to ChatGPT. It handles the communication with ChatGPT, ensuring that the requests are processed and the responses are received correctly.

- **Text2Sign**: The final module takes the text response from ChatGPT and converts it back into sign language animations. This ensures that the communication loop is complete, allowing for smooth and understandable dialogue for ASL users.

<img src="/images/Architecture.jpg" width="600" height="500">

# Installation and Setup

To use this project, clone the repository and install the required libraries listed in `requirements.txt`.

```bash
git clone https://github.com/MassiliaHe/ChatSIGN/
pip install -r requirements.txt
```

### API Key Integration

**Important**: To ensure the application operates correctly, you must obtain a personal API key. Log in to your account on the [OpenAI platform](https://openai.com/api/), generate an API key, and insert this key at line 37 of the `ttos.py` file. This step is crucial for facilitating communication between the application and the ChatGPT API service.

# Usage

Here's how to run the demo using your webcam:

```bash
python main.py
```
