\# Driver Monitoring and Fatigue Alert System



A real-time driver monitoring system developed using Python, Computer Vision, OpenCV, and MediaPipe. The system monitors a driver's eye closure and yawning behavior to identify potential fatigue and generate alerts through local alarms and Telegram notifications.



\## 🚀 Project Overview



Driver fatigue is one of the major factors contributing to road accidents. This project implements a real-time computer vision system that monitors the driver's facial landmarks and detects signs of drowsiness.



The system continuously analyzes webcam frames, calculates Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR), and triggers alerts when predefined conditions are met.



\## ✨ Features



\- Real-time face landmark detection using MediaPipe

\- Eye closure monitoring using Eye Aspect Ratio (EAR)

\- Drowsiness detection using time-based thresholding

\- Yawning detection using Mouth Aspect Ratio (MAR)

\- Real-time webcam monitoring using OpenCV

\- Audio alarm for detected fatigue events

\- Automatic event image capture

\- CSV-based event logging

\- Telegram notifications with captured event images

\- Streamlit dashboard for monitoring and analytics

\- Detection statistics and event history



\## 🛠️ Technologies Used



| Technology | Purpose |

|---|---|

| Python | Application development |

| OpenCV | Webcam capture and image processing |

| MediaPipe | Face landmark detection |

| NumPy | Mathematical calculations |

| Streamlit | Monitoring dashboard |

| Pandas | Event data processing |

| Requests | Telegram API communication |

| CSV | Event logging |



\## 🧠 System Architecture



```text

Webcam Input

&#x20;    |

&#x20;    v

OpenCV Frame Capture

&#x20;    |

&#x20;    v

MediaPipe Face Landmarker

&#x20;    |

&#x20;    v

Facial Landmark Extraction

&#x20;    |

&#x20;    +-------------------+

&#x20;    |                   |

&#x20;    v                   v

Eye Landmark Points   Mouth Landmark Points

&#x20;    |                   |

&#x20;    v                   v

EAR Calculation       MAR Calculation

&#x20;    |                   |

&#x20;    v                   v

Drowsiness Detection  Yawning Detection

&#x20;    |                   |

&#x20;    +---------+---------+

&#x20;              |

&#x20;              v

&#x20;       Event Detection

&#x20;              |

&#x20;    +---------+----------+----------+

&#x20;    |         |          |          |

&#x20;    v         v          v          v

&#x20;  Alarm    Image       CSV       Telegram

&#x20;           Capture     Logging   Notification

&#x20;              |

&#x20;              v

&#x20;      Streamlit Dashboard

```



\## 👁️ Drowsiness Detection



The system calculates the Eye Aspect Ratio (EAR) using facial landmark points around the eyes.



\### Eye Aspect Ratio



EAR measures the relationship between vertical and horizontal distances of the eye.



\\\[

EAR = \\frac{||P\_2-P\_6|| + ||P\_3-P\_5||}{2||P\_1-P\_4||}

\\]



When the eyes are open, EAR is generally higher. When the eyes close, EAR decreases.



\### Detection Configuration



| Parameter | Value |

|---|---|

| EAR Threshold | 0.25 |

| Drowsiness Duration | 1.5 seconds |



\*\*Detection condition:\*\*



When EAR remains below the threshold for the configured duration, the system identifies a drowsiness event.



\## 😴 Yawning Detection



Yawning is detected by measuring the distance between upper and lower mouth landmarks.



\### Mouth Aspect Ratio



\\\[

MAR = \\frac{||P\_2-P\_6|| + ||P\_3-P\_5||}{2||P\_1-P\_4||}

\\]



A higher MAR indicates that the mouth is open.



\### Detection Configuration



| Parameter | Value |

|---|---|

| MAR Threshold | 0.35 |

| Yawning Duration | 1.0 second |



When MAR remains above the configured threshold for the specified duration, the system identifies a yawning event.



\## 🔔 Alert System



When drowsiness or yawning is detected, the system performs the following actions:



1\. Generates an alert.

2\. Plays a local audio alarm.

3\. Captures an image of the event.

4\. Stores the event image in the corresponding directory.

5\. Records event information in a CSV file.

6\. Sends a Telegram notification with the event image.

7\. Updates monitoring statistics and the dashboard.



\## 📊 Dashboard



The Streamlit dashboard provides:



\- Real-time monitoring status

\- Eye Aspect Ratio (EAR)

\- Mouth Aspect Ratio (MAR)

\- Frames per second (FPS)

\- Face detection status

\- Configured detection thresholds

\- Drowsiness event statistics

\- Yawning event statistics

\- Recent event history

\- Event image gallery

\- CSV event log download



\## 📁 Project Structure



```text

Driver-Monitoring-System/

│

├── eye\_monitor.py

├── dashboard.py

├── run\_system.py

├── telegram\_alerts.py

├── test\_telegram.py

├── camera\_test.py

├── face\_detection.py

├── 1\_total\_projectfile.py

│

├── requirements.txt

├── README.md

├── .gitignore

│

├── models/

│   └── face\_landmarker.task

│

├── alerts/

│   ├── drowsiness/

│   └── yawning/

│

├── logs/

│   └── driver\_events.csv

│

└── realtime\_state.json

```



Runtime-generated files and folders such as event images, logs, and monitoring state may not be included in the GitHub repository.



\## ⚙️ Installation



\### 1. Clone the Repository



```bash

git clone https://github.com/swamy-kanda/Driver-Monitoring-System.git

```



\### 2. Navigate to the Project Directory



```bash

cd Driver-Monitoring-System

```



\### 3. Create a Virtual Environment



Windows:



```powershell

python -m venv cv\_env

```



\### 4. Activate the Virtual Environment



```powershell

.\\cv\_env\\Scripts\\Activate.ps1

```



\### 5. Install Dependencies



```bash

pip install -r requirements.txt

```



\## ▶️ Running the System



Run the complete monitoring application:



```bash

python run\_system.py

```



The system uses the webcam for real-time monitoring and the Streamlit dashboard for monitoring statistics and event analytics.



If the application components are configured to run separately:



Start the monitoring system:



```bash

python eye\_monitor.py

```



Start the dashboard:



```bash

streamlit run dashboard.py

```



\## 📱 Telegram Notifications



The project supports Telegram alerts for detected drowsiness and yawning events.



To configure Telegram notifications:



1\. Create a Telegram bot using BotFather.

2\. Obtain the bot token.

3\. Configure your Telegram Chat ID.

4\. Add the credentials to the Telegram alert configuration.

5\. Test the notification functionality.



\*\*Security:\*\* Never publish your Telegram bot token or private credentials in a public repository.



\## 📷 Screenshots



Add screenshots of your project in this section.



Example:



```markdown

!\[Driver Monitoring System](screenshots/monitoring\_system.png)



!\[Dashboard](screenshots/dashboard.png)

```



\## 🎯 Project Objectives



\- Develop a real-time computer vision-based driver monitoring application.

\- Detect prolonged eye closure and yawning behavior.

\- Generate immediate fatigue alerts.

\- Maintain event records for analysis.

\- Provide a graphical dashboard for monitoring system activity.

\- Implement remote notifications using Telegram.



\## 🔮 Future Improvements



\- Head pose estimation

\- Distraction detection

\- Gaze tracking

\- Improved detection under low-light conditions

\- Machine learning-based fatigue classification

\- Edge deployment on embedded Linux platforms

\- Integration with vehicle safety systems



\## ⚠️ Disclaimer



This project is developed for educational and experimental purposes. It is not a certified automotive safety system and should not be relied upon as the sole mechanism for preventing accidents.



\## 👨‍💻 Author



\*\*Swamy Kanda\*\*



GitHub: \[swamy-kanda](https://github.com/swamy-kanda)



Repository: \[Driver Monitoring System](https://github.com/swamy-kanda/Driver-Monitoring-System)



