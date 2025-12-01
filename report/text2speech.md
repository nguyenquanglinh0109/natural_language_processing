# 1. Tổng quan bài toán

## Bài toán
Text-to-Speech (TTS) là công nghệ chuyển đổi văn bản thành âm thanh có giọng tự nhiên. Hệ thống TTS hiện đại không chỉ đọc văn bản mà còn có khả năng điều khiển giọng nói theo cảm xúc, ngữ điệu, âm sắc và phong cách.

## Tình hình nghiên cứu
- TTS phát triển từ các phương pháp dựa trên quy tắc → thống kê → học sâu → diffusion → LLM.
- Nhu cầu ứng dụng ngày càng tăng: trợ lý ảo, audiobook, giáo dục, xe tự hành…
- Gần đây, TTS hướng đến **kiểm soát phong cách tinh vi**, **tổng hợp few-shot/zero-shot**, và **đa ngôn ngữ**.
- Các mô hình mới (Diffusion, LLM-based TTS) giúp giảm lỗi phát âm, tăng tính tự nhiên và khả năng điều khiển.

## Các hướng triển khai chính
- Tổng hợp khớp nối (Articulatory Synthesis)  
- Tổng hợp Formant (Formant Synthesis)  
- Tổng hợp nối đơn vị (Concatenative Synthesis)  
- Tổng hợp tham số thống kê (Statistical Parametric Speech Synthesis – SPSS)  
- Mô hình dựa trên Transformer (FastSpeech, FastPitch)  
- Mô hình dựa trên VAE  
- Mô hình dựa trên Diffusion  
- Mô hình dựa trên LLM  

---

# 2. Ưu – Nhược điểm của từng phương pháp
| Phương pháp | Ưu điểm | Nhược điểm | Phù hợp cho |
|------------|---------|-------------|-------------|
| **Articulatory Synthesis** | - Mô phỏng cơ chế phát âm thực tế<br>- Kiểm soát chi tiết các bộ phận phát âm | - Mô hình hóa rất phức tạp<br>- Âm thanh không tự nhiên | Nghiên cứu ngôn ngữ học, mô phỏng sinh học |
| **Formant Synthesis** | - Dễ kiểm soát tham số<br>- Đa ngôn ngữ, nhẹ | - Giọng robot, thiếu tự nhiên | Thiết bị nhúng, TTS tốc độ cao |
| **Concatenative Synthesis** | - Chất lượng khá tốt (thời tiền deep learning)<br>- Tốc độ nhanh, ổn định | - Không linh hoạt, khó điều khiển cảm xúc<br>- Cần nhiều dữ liệu thu âm chuẩn | Call center, IVR cũ, giọng cố định |
| **SPSS (Statistical Parametric)** | - Kiểm soát pitch, duration tốt<br>- Tốn ít bộ nhớ hơn concatenative | - Âm thanh bị “mờ”, thiếu tự nhiên<br>- Mô hình truyền thống ít linh hoạt | Thiết bị giới hạn tài nguyên, TTS cần điều khiển tham số |
| **Transformer-based (FastSpeech, FastPitch)** | - Tốc độ sinh nhanh<br>- Kiểm soát dễ (độ dài, pitch, năng lượng)<br>- Tự nhiên hơn SPSS | - Cần nhiều dữ liệu<br>- Mô hình acoustic tách biệt vocoder → lỗi lan truyền | Ứng dụng real-time, đa giọng nói |
| **VAE-based TTS** | - Mã hóa không gian latent tốt<br>- Điều khiển phong cách mượt | - Over-smoothing<br>- Khó huấn luyện | TTS đa phong cách, biểu cảm |
| **Diffusion-based TTS** | - Tự nhiên nhất hiện nay<br>- Kiểm soát tốt cảm xúc, style<br>- Giảm lỗi phát âm | - Chậm hơn Transformer<br>- Tốn tài nguyên | TTS chất lượng cao (audiobook, quảng cáo) |
| **LLM-based TTS** | - Hiểu ngữ cảnh<br>- Điều khiển bằng câu tự nhiên (“giọng buồn”, “nhấn mạnh…”)<br>- Zero-shot voice cloning | - Tính toán lớn<br>- Khó tối ưu và yêu cầu dữ liệu lớn | Trợ lý ảo, TTS đa phong cách, hệ thống AI nói tự nhiên |


# 3. Phương pháp tối ưu (pipeline hiện đại)

## 3.1 Giảm nhược điểm – Tối đa ưu điểm bằng pipeline nhiều giai đoạn

### (1) Text Processing
- Chuẩn hóa văn bản (G2P, số đọc thành chữ…)
- Giảm lỗi phát âm ngay từ đầu.

### (2) Acoustic Model (Transformer / Diffusion / LLM)
- Transformer để tăng tốc.  
- Diffusion để tăng độ tự nhiên.  
- LLM để hiểu ngữ cảnh và điều chỉnh style.

### (3) Neural Vocoder (WaveRNN, HiFi-GAN, BigVGAN)
- Khử nhiễu, tái tạo tín hiệu tần số cao.

---

## 3.2 Các chiến lược tối ưu hiện đại

### **A. Hybrid Transformer + Diffusion**
- Transformer dự đoán coarse features → diffusion tinh chỉnh → vừa nhanh vừa tự nhiên.

### **B. LLM to Control TTS**
- LLM điều khiển cảm xúc, style → mô hình acoustic sinh speech theo yêu cầu.

### **C. Few-shot Voice Cloning**
- Extract speaker embedding → fine-tune hoặc sử dụng dựa trên Prompt.

### **D. Multi-scale hoặc hierarchical diffusion**
- Tăng độ sắc nét nhưng giảm số bước sampling.

### **E. Curriculum learning & multi-task training**
- Giảm lỗi pitch, duration, mispronunciation.

### **F. Parallel vocoder (HiFi-GAN, BigVGAN)**
- Xử lý real-time, latency cực thấp.

---

# 4. Reference
1. [Towards Controllable Speech Synthesis in the Era of Large Language
Models: A Systematic Survey](https://arxiv.org/pdf/2412.06602)
2. [A Survey on Neural Speech Synthesis](https://arxiv.org/pdf/2106.15561)