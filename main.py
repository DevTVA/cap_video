import os
import sys
import subprocess
import json
import logging
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
import click
from tqdm import tqdm

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reels_processor.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Load môi trường từ .env nếu có
load_dotenv()

def check_dependencies():
    """Kiểm tra xem ffmpeg và ffprobe đã được cài đặt chưa."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info("FFmpeg và FFprobe đã được cài đặt và sẵn sàng.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Không tìm thấy FFmpeg hoặc FFprobe trên hệ thống của bạn!")
        logger.error("Vui lòng cài đặt FFmpeg và thêm vào biến môi trường PATH.")
        return False

def get_video_metadata(video_path):
    """Lấy thông số metadata của video bằng ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
        data = json.loads(result.stdout)
        
        streams = data.get("streams", [{}])[0]
        format_info = data.get("format", {})
        
        width = int(streams.get("width", 0))
        height = int(streams.get("height", 0))
        duration = float(format_info.get("duration", 0.0))
        
        # Parse frame rate (ví dụ: "30/1" hoặc "24000/1001")
        fps_str = streams.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 30.0
        else:
            fps = float(fps_str)
            
        return {
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps
        }
    except Exception as e:
        logger.error(f"Lỗi khi đọc metadata từ {video_path}: {e}")
        return None

def extract_audio(video_path, audio_path):
    """Trích xuất âm thanh từ video sang định dạng wav 16kHz mono."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi trích xuất âm thanh từ {video_path}: {e}")
        return False

def find_multiple_viral_segments(audio_path, video_duration, segment_len=40, num_segments=5):
    """
    Tìm nhiều phân đoạn viral tiềm năng, sắp xếp theo năng lượng giảm dần,
    đảm bảo các phân đoạn không bị gối đầu (overlap) quá 50% thời lượng.
    """
    if video_duration <= segment_len:
        return [(0.0, video_duration)]
        
    try:
        import librosa
        import numpy as np
        
        logger.info(f"Đang phân tích sóng âm file: {audio_path}")
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Tính RMS Energy trên từng frame
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        
        # Tính số lượng frame tương ứng với 1 giây
        frames_per_sec = sr / hop_length
        # Giới hạn tìm kiếm chỉ trong phần video trước 10 giây cuối cùng để tránh đoạn kết
        search_duration = max(float(segment_len), video_duration - 10.0)
        duration_secs = int(min(len(y) / sr, search_duration))
        
        sec_energy = []
        for i in range(duration_secs):
            start_frame = int(i * frames_per_sec)
            end_frame = int((i + 1) * frames_per_sec)
            if start_frame < len(rms):
                sec_energy.append(np.mean(rms[start_frame:end_frame]))
            else:
                sec_energy.append(0.0)
                
        candidates = []
        sec_energy_temp = list(sec_energy)
        
        for _ in range(num_segments):
            max_energy = -1.0
            best_start = -1
            
            for start in range(0, len(sec_energy_temp) - segment_len + 1):
                window_energy = sum(sec_energy_temp[start:start + segment_len])
                if window_energy > max_energy:
                    max_energy = window_energy
                    best_start = start
                    
            if best_start == -1 or max_energy <= 0:
                break
                
            start_time = float(best_start)
            end_time = min(start_time + segment_len, video_duration)
            candidates.append((start_time, end_time - start_time, max_energy))
            
            # Xóa năng lượng xung quanh phân đoạn đã chọn để tránh trùng lặp gối đầu quá 50%
            clear_start = max(0, best_start - int(segment_len / 2))
            clear_end = min(len(sec_energy_temp), best_start + segment_len)
            for idx in range(clear_start, clear_end):
                sec_energy_temp[idx] = 0.0
                
        # Sắp xếp các ứng viên theo năng lượng giảm dần
        candidates.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Đã tìm thấy {len(candidates)} phân đoạn ứng viên tiềm năng.")
        return [(c[0], c[1]) for c in candidates]
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình phân tích sóng âm: {e}. Tự động trả về phân đoạn mặc định đầu video.")
        return [(0.0, min(float(segment_len), video_duration))]

def split_text_into_lines(text, max_chars=30):
    """
    Chia đoạn text thành các dòng có độ dài tối đa max_chars ký tự,
    không cắt đôi từ.
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_len = len(word)
        addition = word_len + (1 if current_line else 0)
        
        if current_length + addition <= max_chars:
            current_line.append(word)
            current_length += addition
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_len
            
    if current_line:
        lines.append(" ".join(current_line))
        
    return lines

def segment_to_srt_blocks(text, start_time, end_time, max_chars=28):
    """
    Chia một segment của Whisper thành các block srt nhỏ hơn,
    mỗi block tối đa 2 dòng, mỗi dòng tối đa max_chars ký tự.
    Thời gian của các block được chia tỉ lệ theo số ký tự.
    """
    lines = split_text_into_lines(text, max_chars=max_chars)
    if not lines:
        return []
        
    # Gom các dòng thành các block tối đa 2 dòng
    blocks_lines = []
    for i in range(0, len(lines), 2):
        blocks_lines.append(lines[i:i+2])
        
    total_chars = sum(len(line) for line in lines)
    if total_chars == 0:
        return []
        
    duration = end_time - start_time
    blocks = []
    current_start = start_time
    
    for idx, block_line_group in enumerate(blocks_lines):
        block_text = "\n".join(block_line_group)
        block_chars = sum(len(line) for line in block_line_group)
        
        block_duration = duration * (block_chars / total_chars)
        current_end = current_start + block_duration
        
        if idx == len(blocks_lines) - 1:
            current_end = end_time
            
        blocks.append({
            "start": current_start,
            "end": current_end,
            "text": block_text
        })
        current_start = current_end
        
    return blocks

def format_srt_time(seconds):
    """Đổi số giây sang định dạng SRT time: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_subtitles_whisper(audio_path, srt_path, whisper_model="base", device=None):
    """
    Sử dụng OpenAI Whisper local để sinh transcript và lưu file phụ đề .srt
    Đã tối ưu chia nhỏ phụ đề tối đa 2 dòng cân đối.
    """
    try:
        import whisper
        logger.info(f"Đang tải mô hình Whisper '{whisper_model}' (Thiết bị: {device or 'auto'})...")
        model = whisper.load_model(whisper_model, device=device)
        
        logger.info("Đang chạy Whisper Transcribe âm thanh...")
        result = model.transcribe(str(audio_path), verbose=False)
        
        segments = result.get("segments", [])
        
        all_blocks = []
        for seg in segments:
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            if not text:
                continue
                
            # Chia nhỏ segment thành các block srt tối đa 2 dòng, mỗi dòng <= 22 ký tự để cân đối
            blocks = segment_to_srt_blocks(text, start, end, max_chars=22)
            all_blocks.extend(blocks)
            
        logger.info(f"Đã nhận diện và chia thành {len(all_blocks)} block phụ đề cân đối (tối đa 2 dòng). Đang tạo file SRT...")
        
        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, block in enumerate(all_blocks, 1):
                f.write(f"{idx}\n")
                f.write(f"{format_srt_time(block['start'])} --> {format_srt_time(block['end'])}\n")
                f.write(f"{block['text'].upper()}\n\n")
                
        logger.info(f"Đã lưu phụ đề tại: {srt_path}")
        last_subtitle_end = all_blocks[-1]["end"] if all_blocks else 0.0
        return result.get("text", ""), last_subtitle_end
    except ImportError:
        logger.error("Thư viện 'openai-whisper' chưa được cài đặt hoặc import lỗi.")
        return "", 0.0
    except Exception as e:
        logger.error(f"Lỗi trong quá trình sinh phụ đề Whisper: {e}")
        return "", 0.0

def trim_audio(input_audio, output_audio, start_time, duration):
    """Cắt file audio nhanh bằng cách copy codec."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-i", str(input_audio),
        "-t", f"{duration:.3f}",
        "-acodec", "copy",
        str(output_audio)
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        logger.error(f"Lỗi khi cắt audio: {e}")
        return False

def truncate_srt_before_time(srt_path, cut_time):
    """
    Đọc file SRT và loại bỏ tất cả các block phụ đề bắt đầu sau hoặc bằng cut_time,
    hoặc cắt ngắn block nếu nó kéo dài qua cut_time.
    """
    if not os.path.exists(srt_path) or os.path.getsize(srt_path) == 0:
        return
        
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        blocks = content.strip().split("\n\n")
        new_blocks = []
        
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue
                
            # Parse time line (dòng thứ 2)
            time_line = lines[1]
            if "-->" not in time_line:
                continue
                
            start_str, end_str = time_line.split("-->")
            start_str = start_str.strip()
            end_str = end_str.strip()
            
            # Đổi sang số giây
            def parse_time_to_seconds(t_str):
                parts = t_str.replace(",", ".").split(":")
                h = float(parts[0])
                m = float(parts[1])
                s = float(parts[2])
                return h * 3600 + m * 60 + s
                
            start_time = parse_time_to_seconds(start_str)
            end_time = parse_time_to_seconds(end_str)
            
            # Nếu block bắt đầu sau cut_time thì bỏ qua hoàn toàn
            if start_time >= cut_time:
                continue
                
            # Nếu block bắt đầu trước cut_time nhưng kết thúc sau cut_time, cắt ngắn end_time lại
            if end_time > cut_time:
                end_time = cut_time
                # Cập nhật lại dòng thời gian
                def format_srt_time_local(seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    millis = int(round((seconds - int(seconds)) * 1000))
                    if millis >= 1000:
                        millis = 999
                    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
                lines[1] = f"{start_str} --> {format_srt_time_local(end_time)}"
                
            new_blocks.append("\n".join(lines))
            
        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, block in enumerate(new_blocks, 1):
                lines = block.strip().split("\n")
                lines[0] = str(idx) # Đánh số thứ tự lại từ đầu
                f.write("\n".join(lines) + "\n\n")
                
    except Exception as e:
        logger.error(f"Lỗi khi cắt gọn phụ đề: {e}")

def render_final_video(main_video_path, outcard_path, srt_path, output_path, start_time, duration, is_vertical=False):
    """
    Render video trong một bước duy nhất bằng FFmpeg:
    - Định dạng vuông 1080x1080.
    - Kéo hình tràn khung (Scale to fill: phóng to và cắt ở chính giữa).
    - Burn phụ đề in hoa viền đen cân đối (tối đa 2 dòng, nằm trong hình).
    - Đè Out Card (overlay) lên phần cuối của video chính (đúng bằng thời lượng Out Card).
    - Tắt âm thanh video chính và phát âm thanh Out Card ở phần đè lên.
    """
    start_str = f"{start_time:.3f}"
    dur_str = f"{duration:.3f}"
    
    has_outcard = os.path.exists(outcard_path)
    overlay_start = 0.0
    delay_ms = 0
    if has_outcard:
        # Đè Out Card lên video chính trong 3 giây cuối cùng
        outcard_overlay_dur = 3.0
        overlay_start = max(0.0, duration - outcard_overlay_dur)
        delay_ms = int(overlay_start * 1000)
        logger.info(f"Phát hiện Out Card. Sẽ đè Out Card lên 3 giây cuối của video chính (bắt đầu từ giây {overlay_start:.2f}, delay âm thanh: {delay_ms}ms)...")
        
        # Cắt gọn file phụ đề để không phát sinh phụ đề trong khoảng thời gian đè Out Card
        truncate_srt_before_time(srt_path, overlay_start)
        
    # Kiểm tra xem tệp phụ đề có tồn tại và không trống
    has_srt = os.path.exists(srt_path) and os.path.getsize(srt_path) > 0
    
    # Định dạng style phụ đề: MarginV=46 (nằm ở phần dưới hình ảnh), in đậm viền đen
    style = (
        "Fontname=Arial Bold,Fontsize=18,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=3,Alignment=2,MarginV=46"
    )
    
    # Chuẩn bị đường dẫn srt cho FFmpeg filter
    srt_path_esc = str(srt_path).replace("\\", "/").replace(":", "\\:")
    sub_filter = f",subtitles='{srt_path_esc}':force_style='{style}'" if has_srt else ""
    
    # Bộ lọc xử lý video chính: Phóng to tràn khung hình (Scale to Fill: scale to increase + crop)
    main_v_filter = (
        f"scale=1080:1080:force_original_aspect_ratio=increase,"
        f"crop=1080:1080"
        f"{sub_filter},"
        f"fps=30,format=yuv420p,setsar=1/1"
    )
        
    if has_outcard:
        filter_complex = (
            f"[0:v]{main_v_filter}[main_v];"
            f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=enable='gte(t,{overlay_start:.3f})':volume=0[main_a];"
            f"[1:v]scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080,colorkey=0x000000:0.1:0.1,fps=30,format=yuva420p,setsar=1/1,setpts=PTS+{overlay_start:.3f}/TB[out_v];"
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,adelay={delay_ms}|{delay_ms}[out_a];"
            f"[main_v][out_v]overlay=0:0:enable='gte(t,{overlay_start:.3f})'[v];"
            f"[main_a][out_a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", start_str,
            "-i", str(main_video_path),
            "-stream_loop", "-1",
            "-i", str(outcard_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            "-t", dur_str,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "aac",
            "-ar", "44100",
            "-b:a", "192k",
            str(output_path)
        ]
    else:
        filter_complex = (
            f"[0:v]{main_v_filter}[v];"
            f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo[a]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", start_str,
            "-t", dur_str,
            "-i", str(main_video_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "aac",
            "-ar", "44100",
            "-b:a", "192k",
            str(output_path)
        ]
        
    try:
        logger.info(f"Đang tiến hành render video Reels vuông tràn khung 1080x1080 cho: {main_video_path.name}...")
        subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi render video vuông tràn khung: {e}")
        logger.error(f"FFmpeg Render Stderr:\n{e.stderr}")
        return False


def generate_ai_caption(transcript, provider="gemini", api_key=None):
    """
    Đẩy transcript qua API của Gemini hoặc OpenAI để tạo caption Reels thu hút độc giả.
    """
    if not transcript or not transcript.strip():
        logger.warning("Không có transcript để tạo caption.")
        return "Check out this amazing video! #viral #reels"

    prompt = (
        "Read the video transcript and write an English title/caption (strictly 8-12 words) "
        "with relevant emojis for a Facebook Reel. Style: Tabloid, harsh truth, "
        "controversial question, curiosity-inducing. Do not use quotes."
    )
    
    # Ưu tiên lấy API key từ file env hoặc parameter
    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    groq_key = api_key or os.getenv("GROQ_API_KEY")
    
    import time
    
    def try_gemini():
        if not gemini_key:
            return None
        keys = [k.strip() for k in gemini_key.split(",") if k.strip()]
        max_rounds = 3
        for round_idx in range(1, max_rounds + 1):
            if round_idx > 1:
                logger.info(f"Đang thử lại danh sách key Gemini (Lượt {round_idx}/{max_rounds}) sau khi nghỉ 3 giây...")
                time.sleep(3)
                
            for idx, key in enumerate(keys, 1):
                try:
                    import google.generativeai as genai
                    masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
                    logger.info(f"Đang gọi Gemini API với key #{idx} ({masked_key})...")
                    
                    genai.configure(api_key=key)
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    response = model.generate_content([
                        {"role": "user", "parts": [f"System instruction: {prompt}\n\nTranscript: {transcript}"]}
                    ])
                    caption = response.text.strip()
                    caption = caption.replace('"', '').replace("'", "")
                    return caption
                except Exception as e:
                    logger.warning(f"Thử nghiệm Gemini key #{idx} thất bại: {e}")
                    continue
        return None
            
    def try_openai():
        if not openai_key:
            return None
        keys = [k.strip() for k in openai_key.split(",") if k.strip()]
        max_rounds = 3
        for round_idx in range(1, max_rounds + 1):
            if round_idx > 1:
                logger.info(f"Đang thử lại danh sách key OpenAI (Lượt {round_idx}/{max_rounds}) sau khi nghỉ 3 giây...")
                time.sleep(3)
                
            for idx, key in enumerate(keys, 1):
                try:
                    from openai import OpenAI
                    masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
                    logger.info(f"Đang gọi OpenAI API với key #{idx} ({masked_key})...")
                    
                    client = OpenAI(api_key=key)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": transcript}
                        ],
                        max_tokens=60,
                        temperature=0.8
                    )
                    caption = response.choices[0].message.content.strip()
                    caption = caption.replace('"', '').replace("'", "")
                    return caption
                except Exception as e:
                    logger.warning(f"Thử nghiệm OpenAI key #{idx} thất bại: {e}")
                    continue
        return None

    def try_groq():
        if not groq_key:
            return None
        keys = [k.strip() for k in groq_key.split(",") if k.strip()]
        max_rounds = 3
        for round_idx in range(1, max_rounds + 1):
            if round_idx > 1:
                logger.info(f"Đang thử lại danh sách key Groq (Lượt {round_idx}/{max_rounds}) sau khi nghỉ 3 giây...")
                time.sleep(3)
                
            for idx, key in enumerate(keys, 1):
                try:
                    masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
                    logger.info(f"Đang gọi Groq API với key #{idx} ({masked_key})...")
                    
                    import requests
                    headers = {
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": transcript}
                        ],
                        "max_tokens": 60,
                        "temperature": 0.8
                    }
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        json=data,
                        headers=headers,
                        timeout=15
                    )
                    if response.status_code != 200:
                        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
                        
                    response_data = response.json()
                    caption = response_data["choices"][0]["message"]["content"].strip()
                    caption = caption.replace('"', '').replace("'", "")
                    return caption
                except Exception as e:
                    logger.warning(f"Thử nghiệm Groq key #{idx} thất bại: {e}")
                    continue
        return None

    # Xác định thứ tự gọi các API dựa trên cấu hình provider được chọn
    providers_order = []
    if provider == "gemini":
        providers_order = [try_gemini, try_groq, try_openai]
    elif provider == "openai":
        providers_order = [try_openai, try_gemini, try_groq]
    elif provider == "groq":
        providers_order = [try_groq, try_gemini, try_openai]
    else:
        providers_order = [try_gemini, try_groq, try_openai]
        
    for try_func in providers_order:
        res = try_func()
        if res:
            return res
            
    logger.error("TẤT CẢ các API của tất cả nhà cung cấp được cấu hình đều thất bại.")
    return None

def backup_and_cleanup_files(final_video_path, caption_file_path):
    """
    Backup video và caption vào thư mục E:/cap_video_backup/<ngày_hôm_nay>/
    và tự động xóa các thư mục backup cũ hơn 3 ngày trên ổ E.
    """
    import shutil
    import datetime
    
    backup_base = Path("E:/cap_video_backup")
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    today_backup_dir = backup_base / today_str
    
    try:
        # 1. Tạo thư mục backup ngày hôm nay
        today_backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy video và caption vào thư mục backup
        if final_video_path and os.path.exists(final_video_path):
            shutil.copy2(final_video_path, today_backup_dir / final_video_path.name)
            logger.info(f"Đã backup video sang: {today_backup_dir / final_video_path.name}")
            
        if caption_file_path and os.path.exists(caption_file_path):
            shutil.copy2(caption_file_path, today_backup_dir / caption_file_path.name)
            logger.info(f"Đã backup caption sang: {today_backup_dir / caption_file_path.name}")
            
        # 2. Quét và dọn dẹp các thư mục backup cũ hơn 3 ngày
        if backup_base.exists():
            for folder in backup_base.iterdir():
                if folder.is_dir():
                    try:
                        folder_date = datetime.datetime.strptime(folder.name, "%Y-%m-%d").date()
                        days_diff = (datetime.date.today() - folder_date).days
                        if days_diff >= 3:
                            logger.info(f"Phát hiện thư mục backup cũ ({days_diff} ngày trước): {folder.name}. Đang xóa tự động...")
                            shutil.rmtree(folder)
                    except ValueError:
                        # Bỏ qua nếu tên thư mục không phải định dạng YYYY-MM-DD
                        continue
                        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình backup và dọn dẹp dữ liệu: {e}")

def process_single_video(video_path, output_dir, outcard_path, whisper_model, device, segment_duration, api_provider, api_key, output_name=None):
    """Hàm xử lý một video duy nhất, đóng gói try-except để không làm crash cả batch."""
    video_name = video_path.stem
    final_name = output_name if output_name else video_name
    
    try:
        logger.info(f"\n========================================")
        logger.info(f"BẮT ĐẦU XỬ LÝ VIDEO: {video_path.name}")
        logger.info(f"========================================")
        
        # 1. Đọc metadata
        metadata = get_video_metadata(video_path)
        if not metadata:
            logger.error(f"Không thể đọc metadata của {video_path.name}. Bỏ qua video này.")
            return False, None
            
        width, height = metadata["width"], metadata["height"]
        duration = metadata["duration"]
        is_vertical = height >= width
        
        logger.info(f"Thông số gốc: {width}x{height}, Độ dài: {duration:.2f}s, Tỷ lệ: {'Dọc' if is_vertical else 'Ngang'}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            temp_audio = temp_path / f"{video_name}_temp.wav"
            temp_srt = temp_path / f"{video_name}_sub.srt"
            
            # 2. Trích xuất âm thanh gốc
            logger.info("Bước 1: Trích xuất âm thanh từ video gốc...")
            if not extract_audio(video_path, temp_audio):
                raise RuntimeError("Lỗi trích xuất âm thanh.")
                
            # 3. Phân tích tìm các đoạn viral tiềm năng
            logger.info("Bước 2: Tìm các phân đoạn viral tiềm năng...")
            candidates = find_multiple_viral_segments(temp_audio, duration, segment_len=segment_duration, num_segments=5)
            
            best_start = 0.0
            best_duration = duration
            best_srt_path = temp_srt
            transcript = ""
            success = False
            
            # Vòng lặp thử từng phân đoạn đến khi đạt yêu cầu (có phụ đề/tiếng nói nhận diện được)
            for idx, (start_time, duration_to_cut) in enumerate(candidates, 1):
                logger.info(f"Thử nghiệm phân đoạn viral #{idx}: bắt đầu từ {start_time:.1f}s, độ dài {duration_to_cut:.1f}s...")
                
                temp_segment_audio = temp_path / f"{video_name}_segment_{idx}.wav"
                temp_segment_srt = temp_path / f"{video_name}_segment_{idx}.srt"
                
                if not trim_audio(temp_audio, temp_segment_audio, start_time, duration_to_cut):
                    logger.warning(f"Lỗi trích xuất âm thanh cho phân đoạn #{idx}. Thử phân đoạn khác.")
                    continue
                    
                logger.info(f"Chạy Whisper nhận diện phụ đề cho phân đoạn #{idx}...")
                current_transcript, last_subtitle_end = generate_subtitles_whisper(
                    temp_segment_audio, temp_segment_srt, whisper_model=whisper_model, device=device
                )
                
                # Điều kiện đạt yêu cầu: tệp srt tồn tại và không rỗng (chứa phụ đề của câu thoại thực tế)
                if temp_segment_srt.exists() and temp_segment_srt.stat().st_size > 0:
                    logger.info(f"Phân đoạn #{idx} ĐẠT YÊU CẦU (có tiếng nói và phụ đề)! Chọn phân đoạn này.")
                    best_start = start_time
                    # Tự động rút ngắn thời lượng theo câu thoại cuối cùng, cộng thêm 0.5 giây để câu thoại tự nhiên
                    best_duration = min(duration_to_cut, last_subtitle_end + 0.5)
                    logger.info(f"Rút ngắn thời lượng phân đoạn từ {duration_to_cut:.2f}s còn {best_duration:.2f}s (câu thoại cuối kết thúc ở {last_subtitle_end:.2f}s).")
                    best_srt_path = temp_segment_srt
                    transcript = current_transcript
                    success = True
                    break
                else:
                    logger.warning(f"Phân đoạn #{idx} KHÔNG ĐẠT YÊU CẦU (không có tiếng thoại). Đang lặp lại tìm phân đoạn tiếp theo...")
            
            # Fallback nếu không có phân đoạn nào có tiếng nói
            if not success and candidates:
                logger.warning("Không tìm thấy phân đoạn nào chứa tiếng nói. Chọn phân đoạn có năng lượng cao nhất làm mặc định.")
                best_start, best_duration = candidates[0]
                best_srt_path = temp_srt
                with open(best_srt_path, "w", encoding="utf-8") as f:
                    f.write("")
                transcript = ""
                
            # 6. Render video Reels vuông tràn khung tích hợp 1 bước (đè out card 2s cuối)
            logger.info("Bước 5: Render video Reels vuông tràn khung đè Out Card...")
            final_video_name = f"{final_name}.mp4"
            final_video_path = output_dir / final_video_name
            
            if not render_final_video(
                main_video_path=video_path,
                outcard_path=outcard_path,
                srt_path=best_srt_path,
                output_path=final_video_path,
                start_time=best_start,
                duration=best_duration,
                is_vertical=is_vertical
            ):
                raise RuntimeError("Lỗi render video vuông tràn khung.")
                
            # 7. Sinh Tiêu đề/Caption bằng AI từ transcript
            logger.info("Bước 6: Sinh caption AI...")
            caption = None
            if transcript:
                caption = generate_ai_caption(transcript, provider=api_provider, api_key=api_key)
            
            final_caption = caption if caption else "Check out this amazing video! #viral #reels"
            
            logger.info(f"Caption: {final_caption}")
            logger.info(f">>> HOÀN THÀNH XỬ LÝ VIDEO: {video_path.name}")
            logger.info(f">>> File kết quả: {final_video_path}")
            
            # Thực hiện sao lưu dữ liệu và dọn dẹp các thư mục backup cũ hơn 3 ngày trên ổ E (không backup file txt riêng lẻ)
            backup_and_cleanup_files(final_video_path, None)
            
            return True, final_caption
            
    except Exception as e:
        logger.error(f"!!! LỖI khi xử lý video {video_path.name}: {e}")
        return False, None

@click.command()
@click.option("--input-dir", "-i", default="C:/Users/Admin/Desktop/output", help="Thư mục chứa video đầu vào (.mp4, .webm).")
@click.option("--output-dir", "-o", default="C:/Users/Admin/Desktop/output/final_clips", help="Thư mục lưu video đầu ra.")
@click.option("--outcard", "-c", default="E:/ga/mẫu.mp4", help="Đường dẫn file Out Card mẫu (mau.mp4).")
@click.option("--whisper-model", "-w", default="base", type=click.Choice(["tiny", "base", "small", "medium", "large"]), help="Kích thước model Whisper.")
@click.option("--device", "-d", default=None, help="Thiết bị chạy Whisper ('cpu', 'cuda').")
@click.option("--segment-duration", "-t", default=40, type=int, help="Thời lượng đoạn cắt viral (30 đến 45 giây).")
@click.option("--api-provider", default="gemini", type=click.Choice(["gemini", "openai", "groq"]), help="Nhà cung cấp dịch vụ AI để gen caption.")
@click.option("--api-key", default=None, help="API Key của Gemini, OpenAI hoặc Groq (nếu không khai báo trong .env).")
def main(input_dir, output_dir, outcard, whisper_model, device, segment_duration, api_provider, api_key):
    """
    Reels Auto-Cutter CLI: Công cụ chỉnh sửa hàng loạt video tối ưu hóa cho Facebook Reels.
    """
    logger.info("Khởi động Reels Auto-Cutter CLI...")
    
    # Kiểm tra FFmpeg & FFprobe
    if not check_dependencies():
        sys.exit(1)
        
    if not (30 <= segment_duration <= 45):
        logger.error("Độ dài phân đoạn cắt viral phải nằm trong khoảng 30 đến 45 giây.")
        sys.exit(1)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    outcard_path = Path(outcard)
    
    # Tạo các thư mục nếu chưa tồn tại
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Quét các thư mục con trong input_path, loại trừ output_path
    subdirs = [
        d for d in input_path.iterdir()
        if d.is_dir() and d.resolve() != output_path.resolve()
    ]
    
    valid_extensions = [".mp4", ".webm", ".mov", ".mkv"]
    video_tasks = []
    
    # Quét tìm các video trong các thư mục con (đệ quy để tìm các file nằm sâu bên trong)
    if subdirs:
        logger.info(f"Phát hiện các thư mục con trong: {input_path}")
        for subdir in subdirs:
            videos_in_subdir = [
                f for f in subdir.rglob("*")
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]
            if videos_in_subdir:
                # Đặt tên file đầu ra theo tên folder con
                for idx, video_file in enumerate(videos_in_subdir):
                    if len(videos_in_subdir) == 1:
                        out_name = subdir.name
                    else:
                        out_name = f"{subdir.name}_{idx + 1}"
                        
                    if video_file.resolve() == outcard_path.resolve():
                        continue
                    video_tasks.append((video_file, out_name))
                    
    # Fallback: Quét trực tiếp file video trong thư mục gốc nếu không có thư mục con
    if not video_tasks:
        logger.info("Không phát hiện thư mục con chứa video. Tiến hành quét trực tiếp thư mục gốc...")
        video_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        for video_file in video_files:
            if video_file.resolve() == outcard_path.resolve():
                continue
            video_tasks.append((video_file, video_file.stem))
            
    if not video_tasks:
        logger.warning(f"Không tìm thấy video nào để xử lý tại '{input_dir}'!")
        sys.exit(0)
        
    logger.info(f"Tổng số video cần xử lý: {len(video_tasks)}")
    
    # Tự động phát hiện CUDA nếu có cho Whisper
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    logger.info(f"Whisper sẽ chạy trên thiết bị: {device}")
    
    success_count = 0
    fail_count = 0
    results = []
    
    # Xử lý hàng loạt sử dụng tqdm hiển thị progress bar
    for video_file, out_name in tqdm(video_tasks, desc="Đang xử lý video hàng loạt"):
        success, caption = process_single_video(
            video_path=video_file,
            output_dir=output_path,
            outcard_path=outcard_path,
            whisper_model=whisper_model,
            device=device,
            segment_duration=segment_duration,
            api_provider=api_provider,
            api_key=api_key,
            output_name=out_name
        )
        if success:
            success_count += 1
            results.append((out_name, caption))
        else:
            fail_count += 1
            results.append((out_name, "THẤT BẠI (LỖI XỬ LÝ)"))
            
    logger.info("\n========================================")
    logger.info("HOÀN THÀNH QUÁ TRÌNH XỬ LÝ HÀNG LOẠT")
    logger.info(f"Thành công: {success_count}/{len(video_tasks)}")
    if fail_count > 0:
        logger.warning(f"Thất bại: {fail_count}/{len(video_tasks)}")
    logger.info("========================================")
    
    # Ghi file captions.txt tổng hợp vào thư mục output và thư mục backup trên ổ E
    captions_content = ""
    for idx, (video_name, caption) in enumerate(results, 1):
        clean_caption = caption.replace("\n", " ") if caption else ""
        captions_content += f"STT: {idx}\nTên video: {video_name}\nCaption: {clean_caption}\n\n"
        
    # Ghi vào thư mục output
    output_captions_path = output_path / "captions.txt"
    try:
        with open(output_captions_path, "w", encoding="utf-8") as f:
            f.write(captions_content)
        logger.info(f"Đã lưu danh sách caption tổng hợp tại: {output_captions_path}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu file captions.txt tổng hợp: {e}")
        
    # Ghi vào thư mục backup trên ổ E
    import datetime
    backup_base = Path("E:/cap_video_backup")
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    today_backup_dir = backup_base / today_str
    if today_backup_dir.exists():
        backup_captions_path = today_backup_dir / "captions.txt"
        try:
            with open(backup_captions_path, "w", encoding="utf-8") as f:
                f.write(captions_content)
            logger.info(f"Đã backup danh sách caption tổng hợp tại: {backup_captions_path}")
        except Exception as e:
            logger.error(f"Lỗi khi backup file captions.txt: {e}")

    # In bảng tổng hợp caption ở cuối CMD để người dùng dễ sao chép
    print("\n")
    print("=" * 90)
    print(" BẢNG TỔNG HỢP CAPTION VIDEO ".center(90, "="))
    print("=" * 90)
    print(f"{'STT':<5} | {'TÊN VIDEO':<30} | {'CAPTION AI'}")
    print("-" * 90)
    for idx, (video_name, caption) in enumerate(results, 1):
        clean_caption = caption.replace("\n", " ") if caption else ""
        print(f"{idx:<5} | {video_name:<30} | {clean_caption}")
    print("=" * 90)
    print("\n")

if __name__ == "__main__":
    main()
