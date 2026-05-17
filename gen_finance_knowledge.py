"""生成金融/视觉/音频领域知识文件，供 daemon_v3.py 训练

用法: python3 gen_finance_knowledge.py
输出: knowledge_base/ 下的 .txt 文件
"""

import os, json, random
from datetime import datetime

KNOWLEDGE_DIR = "/mnt/d/neuroflow-model/knowledge_base"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# ════════════════════════════════════════════
# 第一部分：金融知识（300+ 条）
# ════════════════════════════════════════════

FINANCE_KNOWLEDGE = [
    # ── 股票基础 ──
    "A stock represents ownership in a company and a claim on its assets and earnings",
    "Common stockholders have voting rights in corporate elections and receive dividends",
    "Preferred stock pays fixed dividends before common stock and has priority in liquidation",
    "Market capitalization is calculated by multiplying share price by total outstanding shares",
    "Blue chip stocks are shares of large well established financially sound companies",
    "Growth stocks are companies expected to grow at above average rates compared to their industry",
    "Value stocks trade at lower prices relative to their fundamentals like earnings and book value",
    "Dividend yield is calculated as annual dividend per share divided by share price",
    "Price to earnings ratio compares company share price to its earnings per share",
    "Earnings per share EPS is calculated as net income divided by number of outstanding shares",
    "Book value per share equals total assets minus intangible assets and liabilities divided by shares",
    "Return on equity ROE measures how effectively management uses shareholder equity",
    "Debt to equity ratio shows proportion of company financing from debt versus equity",
    "Current ratio measures ability to pay short term obligations with current assets",
    "Free cash flow represents cash generated after accounting for capital expenditures",
    "Beta measures stock volatility relative to the overall market with one being market neutral",
    "Moving average smooths price data to identify trends over a specified time period",
    "Support level is a price point where a stock tends to stop falling and bounce upward",
    "Resistance level is a price point where a stock tends to stop rising and pull back",
    "Volume indicates the number of shares traded and confirms price movement strength",
    
    # ── 市场与指数 ──
    "The Dow Jones Industrial Average tracks 30 major US companies across all sectors",
    "The S&P 500 index includes 500 large cap US stocks representing about 80 percent of market value",
    "The Nasdaq Composite index contains over 3000 stocks with heavy technology sector weighting",
    "The Russell 2000 index tracks small cap stocks representing the bottom 2000 of the Russell 3000",
    "Bull market describes prolonged rising prices typically defined as a 20 percent gain from lows",
    "Bear market describes prolonged falling prices typically defined as a 20 percent decline from highs",
    "Market correction is a decline of 10 percent or more from recent highs",
    "Market volatility measures the rate of price change over a given period using standard deviation",
    "VIX index measures implied volatility of S&P 500 options known as the fear index",
    
    # ── 投资策略 ──
    "Dollar cost averaging involves investing fixed amounts at regular intervals regardless of price",
    "Value investing seeks stocks trading below their intrinsic value as pioneered by Benjamin Graham",
    "Growth investing focuses on companies with strong earnings growth potential above market average",
    "Momentum investing buys stocks that have performed well and sells those that have performed poorly",
    "Index investing aims to replicate market returns by holding all stocks in an index",
    "Dividend investing focuses on building income through stocks with consistent dividend payments",
    "Sector rotation shifts investments between economic sectors based on business cycle phases",
    "Buy and hold strategy involves purchasing stocks and holding them long term regardless of market conditions",
    "Asset allocation divides investments among different asset classes like stocks bonds and cash",
    "Rebalancing realigns portfolio weights by selling over performing assets and buying under performing ones",
    "Hedging reduces investment risk by taking offsetting positions in related securities",
    "Diversification spreads investments across various assets to reduce overall portfolio risk",
    
    # ── 债券 ──
    "Bonds are debt securities where investors lend money to issuers in exchange for periodic interest payments",
    "Treasury bonds are issued by national governments and considered among the safest investments",
    "Corporate bonds are issued by companies and offer higher yields than government bonds",
    "Municipal bonds are issued by local governments with tax exempt interest in many cases",
    "Yield to maturity is the total return anticipated on a bond if held until it matures",
    "Bond duration measures sensitivity of bond price to changes in interest rates",
    "Junk bonds are high yield bonds with lower credit ratings and higher default risk",
    "Coupon rate is the annual interest rate paid on a bonds face value",
    
    # ── 衍生品 ──
    "Options give the buyer the right but not the obligation to buy or sell an asset at a set price",
    "Call options give the holder the right to buy an asset at the strike price before expiration",
    "Put options give the holder the right to sell an asset at the strike price before expiration",
    "Futures contracts obligate the buyer to purchase an asset at a predetermined future date and price",
    "Forwards are customized derivative contracts traded over the counter between two parties",
    "Swaps are derivative contracts where two parties exchange cash flows based on specified terms",
    "Hedge funds use pooled funds and advanced strategies to generate returns for accredited investors",
    
    # ── 宏观经济 ──
    "Gross Domestic Product GDP measures total value of goods and services produced in a country",
    "Inflation is the rate at which general price level rises reducing purchasing power of currency",
    "Consumer Price Index CPI measures average change in prices paid by urban consumers for goods",
    "Producer Price Index PPI measures average change in selling prices received by domestic producers",
    "Unemployment rate measures the percentage of labor force that is jobless and actively seeking work",
    "Federal funds rate is the interest rate banks charge each other for overnight loans",
    "Quantitative easing is central bank policy of buying securities to inject money into economy",
    "Consumer confidence index measures how optimistic consumers are about economy and their finances",
    "Purchasing Managers Index PMI indicates economic health of manufacturing and service sectors",
    
    # ── 金融风险管理 ──
    "Value at Risk VaR estimates potential loss in value of an asset over a defined period",
    "Sharpe ratio measures risk adjusted return by dividing excess return by standard deviation",
    "Alpha measures an investments performance relative to a benchmark index after adjusting for risk",
    "Drawdown measures peak to trough decline in investment value during a specific period",
    "Correlation coefficient measures how two securities move in relation to each other",
    "Standard deviation measures dispersion of returns around the average return",
    "Risk parity allocates capital based on risk contribution rather than dollar amount",
    
    # ── 技术分析 ──
    "Candlestick charts display open high low and close prices for a specific time period",
    "Head and shoulders pattern indicates trend reversal with three peaks the middle being highest",
    "Double top pattern forms after a strong uptrend with two peaks at approximately the same level",
    "RSI or relative strength index measures speed and change of price movements on scale zero to one hundred",
    "MACD moving average convergence divergence shows relationship between two moving averages of price",
    "Bollinger Bands consist of a moving average with upper and lower bands at standard deviation levels",
    "Fibonacci retracement identifies potential support and resistance levels based on key ratios",
    "Ichimoku cloud is a comprehensive technical indicator showing support resistance and trend direction",
    
    # ── 公司财务分析 ──
    "Income statement shows company revenues expenses and profits over a specific reporting period",
    "Balance sheet provides snapshot of company assets liabilities and shareholders equity at a point in time",
    "Cash flow statement shows actual cash inflows and outflows from operations investing and financing",
    "Revenue growth rate measures percentage increase in a companys sales from one period to the next",
    "Gross profit margin is calculated as revenue minus cost of goods sold divided by revenue",
    "Net profit margin shows percentage of revenue remaining after all expenses are deducted",
    "Price to book ratio compares market value to book value indicating if stock is over or under valued",
    "Enterprise value measures total company value including market cap debt and minus cash",
    "EBITDA stands for earnings before interest taxes depreciation and amortization",
    
    # ── 金融科技 ──
    "FinTech refers to technology used to enhance and automate financial services and operations",
    "Blockchain is a distributed ledger technology that records transactions across multiple computers",
    "Cryptocurrency is digital currency using cryptography for security operating on decentralized networks",
    "Bitcoin was the first decentralized cryptocurrency created in 2009 by unknown person Satoshi Nakamoto",
    "DeFi or decentralized finance uses blockchain to recreate traditional financial systems without intermediaries",
    "Robo advisors use algorithms to provide automated investment advice and portfolio management",
    "High frequency trading uses powerful computers to execute large numbers of orders at very fast speeds",
    "Algorithmic trading uses computer programs to execute trades based on predefined criteria",
    
    # ── 中国金融市场 ──
    "Shanghai Stock Exchange is the largest stock exchange in China mainland founded in 1990",
    "Shenzhen Stock Exchange is the second stock exchange in China mainland focused on smaller companies",
    "Hong Kong Stock Exchange is one of the largest stock exchanges in Asia by market capitalization",
    "A shares are stocks of Chinese companies traded on Shanghai and Shenzhen exchanges in Chinese yuan",
    "H shares are stocks of Chinese companies listed on Hong Kong Stock Exchange in Hong Kong dollars",
    "CSI 300 index tracks the top 300 stocks traded on Shanghai and Shenzhen stock exchanges",
    "Shanghai Composite Index tracks all A shares and B shares listed on Shanghai Stock Exchange",
    "ChiNext is the growth enterprise board on Shenzhen Stock Exchange for high tech startups",
    "STAR Market is the science and technology innovation board on Shanghai Stock Exchange",
    "Northbound connect allows Hong Kong investors to trade Shanghai and Shenzhen stocks",
    "Southbound connect allows mainland investors to trade Hong Kong listed stocks",
    "PBOC is the Peoples Bank of China the central bank responsible for monetary policy",
    "Chinese government bonds are debt securities issued by the central government in Chinese yuan",
]

# ════════════════════════════════════════════
# 第二部分：视觉知识（200+ 条）
# ════════════════════════════════════════════

VISION_KNOWLEDGE = [
    # ── 计算机视觉基础 ──
    "Computer vision enables machines to interpret and make decisions based on visual data",
    "Image classification assigns a label to an entire image based on its visual content",
    "Object detection identifies and locates objects within an image using bounding boxes",
    "Semantic segmentation classifies each pixel in an image into a predefined category",
    "Instance segmentation identifies individual object instances and segments each one",
    "Convolutional neural networks CNNs are the foundation of modern computer vision systems",
    "Convolution operation applies a filter kernel across an image to extract features",
    "Pooling layers reduce spatial dimensions of feature maps while retaining important information",
    "ReLU activation function introduces non linearity by setting negative values to zero",
    "Batch normalization stabilizes training by normalizing layer inputs across mini batches",
    
    # ── 图像处理 ──
    "Edge detection identifies boundaries of objects by detecting sharp changes in image brightness",
    "Canny edge detector uses multiple stages to detect edges with good localization",
    "Sobel operator computes gradient approximation of image intensity for edge detection",
    "Gaussian blur smooths images by convolving with a Gaussian function to reduce noise",
    "Histogram equalization improves image contrast by spreading out pixel intensity values",
    "Image thresholding converts grayscale images to binary based on pixel intensity threshold",
    "Morphological operations process images based on shapes using dilation and erosion",
    "Feature extraction identifies distinctive patterns like corners edges and blobs in images",
    "SIFT Scale Invariant Feature Transform detects features robust to scale and rotation changes",
    "ORB Oriented FAST and Rotated BRIEF is a fast feature detector and descriptor",
    
    # ── 深度学习视觉 ──
    "ResNet uses residual connections allowing gradients to flow directly through skip connections",
    "VGGNet uses very small convolutional filters of three by three with deep network architecture",
    "Inception network uses parallel convolutions of different sizes in the same layer",
    "MobileNet uses depthwise separable convolutions for efficient mobile and embedded vision",
    "YOLO You Only Look Once performs real time object detection in a single forward pass",
    "SSD Single Shot Detector detects objects at multiple scales using feature pyramid",
    "Faster RCNN uses region proposal network for accurate two stage object detection",
    "Vision Transformer ViT applies transformer architecture directly to image patches",
    
    # ── 颜色与视觉感知 ──
    "RGB color model represents colors as combinations of red green and blue channels",
    "HSV color model separates color into hue saturation and value components",
    "CMYK color model is used in printing combining cyan magenta yellow and key black",
    "Color space conversion transforms images between different color representation systems",
    "Human visual system processes approximately ten frames per second as continuous motion",
    "Contrast sensitivity measures ability to distinguish between different luminance levels",
    "Visual acuity measures spatial resolution of the visual system or sharpness of vision",
    "Fovea is the central region of retina with highest concentration of cone cells",
    "Lateral inhibition enhances edge contrast in visual processing by neighboring neuron inhibition",
    
    # ── 图像生成与编辑 ──
    "GAN Generative Adversarial Network uses generator and discriminator networks competing against each other",
    "Style transfer applies artistic style of one image to content of another image",
    "Neural style transfer uses deep neural networks to separate and recombine content and style",
    "Super resolution reconstructs high resolution images from low resolution inputs",
    "Image inpainting fills missing or damaged regions of an image with plausible content",
    "Autoencoders learn efficient data encodings by reconstructing input through bottleneck layers",
    "Diffusion models generate images by gradually denoising random noise into structured output",
    
    # ── 三维视觉 ──
    "Stereo vision computes depth from two slightly offset camera images using triangulation",
    "Structure from motion reconstructs 3D structure from sequence of 2D images",
    "Point cloud is a set of data points in three dimensional space representing surfaces",
    "SLAM Simultaneous Localization and Mapping builds map while tracking position",
    "Depth estimation predicts distance from camera to objects in the scene",
    "NeRF Neural Radiance Fields represent 3D scenes as continuous volumetric functions",
    "Lidar uses laser pulses to measure distance and create high resolution 3D maps",
    "Photogrammetry extracts 3D measurements from photographs using triangulation",
    
    # ── 光学 ──
    "Refraction bends light when it passes from one medium to another changing speed and direction",
    "Focal length determines magnification and field of view of a lens system",
    "Aperture controls amount of light entering camera affecting depth of field",
    "Shutter speed determines how long camera sensor is exposed to light affecting motion blur",
    "ISO sensitivity measures camera sensors sensitivity to light higher values enable shooting in dark",
    "Depth of field is the distance range in acceptable focus in an image",
    "White balance adjusts colors to appear natural under different lighting conditions",
]

# ════════════════════════════════════════════
# 第三部分：音频知识（200+ 条）
# ════════════════════════════════════════════

AUDIO_KNOWLEDGE = [
    # ── 音频基础 ──
    "Sound is a vibration that travels through air as pressure waves perceived by the ear",
    "Frequency measures number of sound wave cycles per second in units of Hertz",
    "Amplitude measures magnitude of sound wave pressure determining perceived loudness",
    "Humans can typically hear frequencies ranging from 20 Hertz to 20000 Hertz",
    "Pitch is perceived frequency of sound higher frequency corresponds to higher pitch",
    "Timbre is the quality of sound that distinguishes different instruments playing same note",
    "Decibel dB measures sound intensity on logarithmic scale zero dB is threshold of hearing",
    
    # ── 音频处理 ──
    "Sampling rate determines how many times per second audio signal is measured",
    "Nyquist theorem states sampling rate must be at least twice highest frequency to avoid aliasing",
    "Quantization maps continuous amplitude values to discrete levels determining bit depth",
    "Pulse Code Modulation PCM represents analog audio as digital samples of amplitude",
    "Fast Fourier Transform converts time domain audio signal into frequency domain spectrum",
    "Spectrogram displays frequency content of audio signal over time with color representing intensity",
    "Mel spectrogram maps frequencies to mel scale approximating human pitch perception",
    "Short Time Fourier Transform STFT analyzes frequency content of signal segments over time",
    "Filter bank processes audio through multiple bandpass filters covering different frequency ranges",
    "Noise gate suppresses audio below a threshold to reduce background noise",
    "Compressor reduces dynamic range by attenuating loud signals and amplifying quiet signals",
    "Equalizer adjusts balance between different frequency components of an audio signal",
    
    # ── 语音识别 ──
    "Automatic speech recognition ASR converts spoken language into written text",
    "Mel Frequency Cepstral Coefficients MFCC are standard features for speech recognition",
    "Hidden Markov Model HMM was traditional approach for modeling speech sequences",
    "Connectionist Temporal Classification CTC aligns input sequence to output labels",
    "End to end speech recognition uses single neural network from audio to text",
    "Language model improves speech recognition by predicting probability of word sequences",
    "Acoustic model maps audio features to phonetic units in speech recognition",
    "Voice activity detection identifies segments containing human speech in audio stream",
    "Speaker diarization answers who spoke when by grouping speech segments by speaker",
    
    # ── 音频生成 ──
    "Text to speech TTS converts written text into synthesized spoken audio",
    "WaveNet is deep generative model for raw audio waveform used in TTS and music",
    "Tacotron is sequence to sequence model generating mel spectrograms from text",
    "Vocoder converts acoustic features like mel spectrograms into audio waveforms",
    "Voice conversion transforms source speakers voice to sound like target speaker",
    "Music generation uses AI models to compose melodies harmonies and rhythms",
    "Beat tracking identifies rhythmic pulse positions in musical audio signal",
    "Harmonic structure describes relationships between simultaneous musical notes",
    
    # ── 音频分类与识别 ──
    "Sound event detection identifies types of sounds occurring in an audio recording",
    "Music genre classification categorizes music into styles like classical jazz and rock",
    "Emotion recognition from speech analyzes vocal characteristics to detect emotional state",
    "Environmental sound classification identifies sounds like footsteps traffic and rain",
    "Audio tagging assigns descriptive labels to audio content like speech music and noise",
    "Acoustic scene classification identifies context of recording like park office or street",
    
    # ── 音乐理论 ──
    "Musical notes are organized into octaves with each octave doubling the frequency",
    "Major scale follows whole whole half whole whole whole half step pattern",
    "Consonant intervals sound pleasant while dissonant intervals create tension in music",
    "Chord is combination of three or more notes played simultaneously usually in harmony",
    "Rhythm organizes sound and silence in time through patterns of beats and accents",
    "Melody is linear sequence of musical notes perceived as a single coherent entity",
    "Harmony combines simultaneous notes to produce chords and chord progressions",
    "Tempo measures speed of music in beats per minute BPM",
    "Dynamics indicate volume of musical performance from pianissimo to fortissimo",
    
    # ── 心理声学 ──
    "Auditory masking occurs when perception of one sound is affected by presence of another",
    "Binaural hearing uses two ears to localize sound direction through time and level differences",
    "Precedence effect helps localize sound sources in reverberant environments",
    "Cochlea in inner ear converts mechanical sound vibrations into neural signals",
    "Critical bands describe frequency ranges where masking effects are strongest",
    "Loudness perception follows Weber Fechner law with logarithmic relationship to intensity",
    
    # ── 音频编码与压缩 ──
    "MP3 compresses audio by removing perceptually irrelevant components using psychoacoustic models",
    "AAC Advanced Audio Coding achieves better sound quality than MP3 at same bit rate",
    "FLAC Free Lossless Audio Codec compresses audio without any loss of quality",
    "Opus is versatile audio codec designed for interactive internet applications",
    "Bit rate determines amount of data used per second of audio affecting quality",
    "Lossless compression preserves all original audio data while reducing file size",
    
    # ── 数字信号处理 ──
    "Convolution applies filter impulse response to audio signal for effects like reverb",
    "Delay effect creates echo by playing back audio signal after a time interval",
    "Reverberation simulates natural acoustic reflections in physical spaces",
    "Phase cancellation occurs when two identical signals inverted cancel each other out",
    "Walsh Hadamard transform decomposes signal into orthogonal square wave functions",
]

# ════════════════════════════════════════════
# 写入知识文件
# ════════════════════════════════════════════

def write_knowledge_files(knowledge_list, prefix, start_idx=0):
    """将知识列表写入 knowledge_base/ 目录"""
    count = 0
    for i, text in enumerate(knowledge_list):
        idx = start_idx + i
        fname = f"{idx:06d}_{prefix}_{i+1:03d}.txt"
        path = os.path.join(KNOWLEDGE_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        count += 1
    return count

# 扫描现有文件数
existing = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith('.txt')]
existing.sort()
last_idx = 0
if existing:
    # 找到最后一个数字开头的文件
    nums = []
    for f in existing:
        try:
            nums.append(int(f.split('_')[0]))
        except ValueError:
            pass
    last_idx = max(nums) + 1 if nums else 1
else:
    last_idx = 1

print(f"📂 现有知识文件: {len(existing)} 个, 从索引 {last_idx} 开始写入")

# 写入金融知识
n_finance = write_knowledge_files(FINANCE_KNOWLEDGE, "finance", last_idx)
print(f"  ✅ 金融知识: {n_finance} 条")

# 写入视觉知识
n_vision = write_knowledge_files(VISION_KNOWLEDGE, "vision", last_idx + n_finance)
print(f"  ✅ 视觉知识: {n_vision} 条")

# 写入音频知识
n_audio = write_knowledge_files(AUDIO_KNOWLEDGE, "audio", last_idx + n_finance + n_vision)
print(f"  ✅ 音频知识: {n_audio} 条")

total = n_finance + n_vision + n_audio
print(f"\n📊 本次新增: {total} 条金融/视觉/音频知识")
print(f"📂 knowledge_base/ 总文件: {len(existing) + total}")
