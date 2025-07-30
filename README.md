# Tourist Satisfaction Analyzer

Aplikasi web untuk menganalisis tingkat kepuasan wisatawan berdasarkan review dan rating menggunakan Machine Learning dan Natural Language Processing.

## 🌟 Features

- 📊 **Dashboard Interaktif** - Visualisasi data real-time dengan charts dan statistik
- 🔍 **Analisis Sentimen** - Analisis sentimen otomatis dari review wisatawan
- 📈 **Rekomendasi Prioritas** - Saran perbaikan berdasarkan analisis data
- 📁 **Upload Data** - Support Excel dan CSV file upload
- 📱 **Responsive Design** - Optimized untuk desktop dan mobile
- 🤖 **Prediksi Rating** - Machine learning untuk prediksi rating dari review

## 🚀 Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** Bootstrap 5, HTML5, CSS3, JavaScript
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, TextBlob
- **Visualization:** Plotly.js, Chart.js
- **Deployment:** Render

## 📋 Data Format

Upload file Excel/CSV dengan kolom berikut:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| wisata | string | Nama tempat wisata | "Pantai Kuta" |
| rating | integer | Rating 1-5 | 4 |
| review | string | Text review | "Tempat yang bagus" |
| waktu_kunjungan | string | Waktu kunjungan | "Pagi" |

## 🔧 Local Development

1. **Clone repository:**
```bash
git clone https://github.com/username/tourist-satisfaction-analyzer.git
cd tourist-satisfaction-analyzer

Create virtual environment:
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

Install dependencies:
pip install -r requirements.txt

Run application:
python app.py

Open browser: http://localhost:5000

🌐 Live Demo

Deployed on Render: https://your-app-name.onrender.com

📊 Analysis Features

Dashboard
Total reviews dan rating rata-rata
Tingkat kepuasan wisatawan
Distribusi rating dan sentimen
Top destinasi wisata
Analisis waktu kunjungan

Deep Analysis
Analisis keluhan utama
Keywords positif dan negatif
Rekomendasi prioritas perbaikan
Analisis berdasarkan waktu kunjungan
Detail per destinasi wisata

Machine Learning
Sentiment analysis menggunakan NLP
Rating prediction dari text review
Keyword extraction dan clustering
Statistical analysis dan correlation

🤝 Contributing

Fork the repository
Create feature branch (git checkout -b feature/AmazingFeature)
Commit changes (git commit -m 'Add some AmazingFeature')
Push to branch (git push origin feature/AmazingFeature)
Open a Pull Request

📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author

Created with ❤️ by [Your Name]

🙏 Acknowledgments

Bootstrap team for the amazing CSS framework
Plotly team for interactive visualizations
Flask community for the excellent documentation
Render for free hosting platform