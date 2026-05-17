"""
NeuroFlow 自主学习批处理 v3 — TrainableHead 在线进化版
每2分钟运行一次：联网获取知识 → 推理 → SGD训练可训练决策层
数据源：HackerNews API, GitHub API, NPR RSS, 本地+中文知识库

v3 改动 (vs v2):
  旧: SelfEvolution 噪声扰动（随机变异 → 无方向性）
  新: TrainableHead SGD（交叉熵+价值预测 → 有梯度的在线学习）
  每次推理后立即训练：target=自蒸馏决策, reward=推理置信度
"""
import sys, os, time, json, random, numpy as np
from datetime import datetime
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

sys.path.insert(0, "/mnt/d/neuroflow-model")
from neuroflow._core import create_multimodal
from neuroflow.cognition import NeuroSymbolicReasoner
from neuroflow.trainable_head import TrainableHead

# === 配置 ===
STATE_FILE = "/mnt/d/neuroflow-model/daemon_state.json"
WEIGHTS_FILE = "/home/administrator/.hermes/neuroflow_weights.npz"
WEIGHTS_FILE_V4 = "/home/administrator/.hermes/neuroflow_weights_v4.npz"  # 新维度权重
KB_DIR = "/mnt/d/neuroflow-model/knowledge_base"
BATCH_SIZE = 200
HTTP_TIMEOUT = 8
SGD_EPOCHS = 15        # 每轮5→15，epoch 0 全推理 + epoch 1+ 噪声复用
NOISE_STD = 0.03        # epoch 2+ 编码向量高斯噪声标准差
REWARD_NOISE = 0.05     # epoch 2+ 置信度噪声

# 硬件优化参数 — 双路 Xeon E5-2666 v3 ×2, 40T, 64GB
TEXT_DIM = 768
HIDDEN_DIM = 512
OUTPUT_DIM = 16
MEMORY_DIM = 256
NUM_LAYERS = 4
HEAD_HIDDEN = 512
HEAD_ACTIONS = 16
OMP_NUM_THREADS = "40"   # 全核心

os.makedirs(KB_DIR, exist_ok=True)
os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS


# ============================================================
# 联网获取知识（不变）
# ============================================================

def fetch_hn_top_stories(limit=20):
    endpoints = [
        ("topstories", 30), ("newstories", 20), ("beststories", 15),
        ("askstories", 10), ("showstories", 10)
    ]
    all_ids = set()
    for ep, n in random.sample(endpoints, min(3, len(endpoints))):
        try:
            req = urllib.request.Request(
                f"https://hacker-news.firebaseio.com/v0/{ep}.json",
                headers={"User-Agent": "NeuroFlow/1.0"}
            )
            ids = json.loads(urllib.request.urlopen(req, timeout=HTTP_TIMEOUT).read())
            all_ids.update(ids[:n])
        except:
            continue

    seen_file = "/mnt/d/neuroflow-model/hn_seen.json"
    seen = set()
    if os.path.exists(seen_file):
        try:
            with open(seen_file) as f:
                seen = set(json.load(f))
        except:
            pass

    items = []
    for sid in list(all_ids)[:limit*2]:
        if sid in seen:
            continue
        try:
            req2 = urllib.request.Request(
                f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                headers={"User-Agent": "NeuroFlow/1.0"}
            )
            story = json.loads(urllib.request.urlopen(req2, timeout=HTTP_TIMEOUT).read())
            title = story.get("title", "")
            if title and title not in items:
                items.append(title)
                seen.add(sid)
            if len(items) >= limit:
                break
        except:
            continue

    with open(seen_file, "w") as f:
        json.dump(list(seen)[-500:], f)
    return items


def fetch_github_trending(limit=10):
    rate_file = "/mnt/d/neuroflow-model/gh_rate.json"
    now = time.time()
    rate_info = {"last_fetch": 0, "calls_this_hour": 0}
    if os.path.exists(rate_file):
        try:
            with open(rate_file) as f:
                saved = json.load(f)
            if now - saved.get("last_fetch", 0) > 3600:
                rate_info = {"last_fetch": now, "calls_this_hour": 0}
            else:
                rate_info = saved
        except:
            pass

    if rate_info.get("calls_this_hour", 0) >= 10:
        return []

    repos = [
        "torvalds/linux", "microsoft/vscode", "facebook/react",
        "tensorflow/tensorflow", "kubernetes/kubernetes", "apple/swift",
        "rust-lang/rust", "golang/go", "ansible/ansible",
        "home-assistant/core", "yt-dlp/yt-dlp", "microsoft/TypeScript",
        "vercel/next.js", "django/django", "pytorch/pytorch"
    ]
    items = []
    max_repos = min(limit, 2, len(repos))
    for repo in random.sample(repos, max_repos):
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{repo}",
                headers={"User-Agent": "NeuroFlow/1.0", "Accept": "application/vnd.github.v3+json"}
            )
            data = json.loads(urllib.request.urlopen(req, timeout=HTTP_TIMEOUT).read())
            rate_info["calls_this_hour"] = rate_info.get("calls_this_hour", 0) + 1
            rate_info["last_fetch"] = now
            desc = data.get("description", "")
            if desc and len(desc) > 10:
                items.append(f"{repo.split('/')[1]}: {desc}")
            else:
                items.append(f"Open source project {repo}: {data.get('stargazers_count', 0)} stars")
        except urllib.error.HTTPError as e:
            if e.code == 403:
                rate_info["calls_this_hour"] = 999
                break
            continue
        except:
            continue

    with open(rate_file, "w") as f:
        json.dump(rate_info, f)
    return items


def fetch_npr_headlines(limit=10):
    try:
        req = urllib.request.Request(
            "https://feeds.npr.org/1001/rss.xml",
            headers={"User-Agent": "NeuroFlow/1.0"}
        )
        data = urllib.request.urlopen(req, timeout=HTTP_TIMEOUT).read()
        root = ET.fromstring(data)
        items = []
        for item in root.findall('.//item')[:limit]:
            title = item.find('title')
            if title is not None and title.text:
                items.append(title.text.strip())
        return items
    except:
        return []


# ============================================================
# 本地知识库
# ============================================================
LOCAL_KB = [
    "Electromagnetic waves include radio waves microwaves infrared visible light ultraviolet X-rays",
    "The Roman Empire dominated the Mediterranean world for over five hundred years",
    "Gross domestic product measures the total value of goods and services produced",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight",
    "The water cycle involves evaporation condensation precipitation and collection",
    "The Pythagorean theorem states that the square of the hypotenuse equals sum of squares",
    "Cells are the basic structural and functional units of all living organisms",
    "Homeostasis maintains stable internal conditions despite external environmental changes",
    "The Renaissance was a period of cultural rebirth in Europe",
    "The immune system protects the body against pathogens through specialized cells",
    "Maslow hierarchy of needs ranges from physiological requirements to self actualization",
    "Supply and demand determine market prices in a competitive economy",
    "Entropy measures the disorder in a system and always increases in isolated systems",
    "Atoms consist of protons neutrons and electrons orbiting the nucleus",
    "Genes are segments of DNA that code for proteins and determine hereditary traits",
    "The scientific method involves observation hypothesis testing and peer review",
    "Algorithms are step by step procedures for solving computational problems",
    "Database systems organize and retrieve structured data efficiently using SQL",
    "Ethics examines moral principles that govern behavior and decision making",
    "Linear algebra deals with vectors matrices and systems of linear equations",
    "Operant conditioning uses rewards and punishments to shape behavior",
    "The Fibonacci sequence appears in nature from sunflower seeds to galaxy spirals",
    "Enzymes are biological catalysts that speed up chemical reactions",
    "DNA is a double helix structure containing genetic information for all life",
    "Prime numbers are integers greater than one divisible only by themselves and one",
    "The periodic table organizes elements by atomic number and chemical properties",
    "Complex numbers consist of real and imaginary parts represented as a plus bi",
    "Cryptography protects information through encryption and decryption techniques",
    "Existentialism emphasizes individual freedom choice and the search for meaning",
    "The Industrial Revolution began in Britain and mechanized production processes",
    "Shakespeare wrote tragedies comedies and histories exploring human nature",
    "Cloud computing provides on demand access to computing resources over the internet",
    "The speed of light in vacuum is approximately three hundred thousand kilometers per second",
    "Memory involves encoding storage and retrieval of information in the brain",
    "Newton laws of motion describe the relationship between force mass and acceleration",
    "Stoicism teaches that virtue is the highest good and we should accept what we cannot control",
    "Mitochondria are the powerhouses of the cell producing ATP through respiration",
    "The human brain contains approximately eighty six billion neurons",
    "Quantum mechanics describes the behavior of particles at atomic scales",
    "Classical conditioning associates a neutral stimulus with a reflexive response",
    "Stoicism teaches virtue is the highest good",
    "RTX 5090 and M4 MacBook Air: Can It Game?",
    "Why Your AI Can Write a Novel but Still Struggles to Count to Fifty LLMHall",
    "Princeton mandates proctoring for in-person exams, upending 133 year precedent",
    "New Fragnesia Linux flaw lets attackers gain root privileges",
    "'Millions' of pounds saved by replacing Palantir tech in refugee system",
    "Show HN: JDS – a Copilot skill suite for structuring AI coding behavior",
    "Accelerating Hamming Quasi-Cyclic (HQC) with Additive FFT",
    "Honda posts first annual loss on $9B EV writedown, scraps EV sales goals",
    "react: The library for web and native user interfaces.",
    "go: The Go programming language",
    "June 9. Dead man switch and MS RCE drops promised?",
    "The 10 best songs competing at (a very contentious) Eurovision",
    "Show HN: SwiftUI package for onboarding flows in iOS apps",
    "Claude Code Issue that important facts were forgotten when sessions were reseted",
    "Porting 3D Movie Maker to Linux",
    "The Power of a Free Popsicle (2018)",
    "linux: Linux kernel source tree",
    "swift: The Swift Programming Language",
    "kubernetes: Production-Grade Container Scheduling and Management",
    "pytorch: Tensors and Dynamic neural networks in Python with strong GPU acceleration",
    "yt-dlp: A feature-rich command-line audio/video downloader",
    "ansible: Ansible is a radically simple IT automation platform that makes your applications and systems easier to deploy and maintain. Automate everything from code deployment to network configuration ",
    "Velonus – Open-source AppSec scanner that deduplicates SAST noise",
    "7 in 10 Americans oppose data centers being built in their communities",
    "django: The Web framework for perfectionists with deadlines.",
    "CSS Rhythmic Sizing Module Level 1",
    "TypeScript: TypeScript is a superset of JavaScript that compiles to clean JavaScript output.",
    "We Tested DeepSeek V4 Pro and Flash Against Claude Opus 4.7 and Kimi K2.6",
    "Show HN: Latencies and BEIR – Typesense, Meilisearch, Elasticsearch, Amgix Now",
    "Find vendors used by any company",
    "GPT convinced me there was a bug in my code before a freeze",
    "I resurrected the web from the past and it got weird",
    "The Efficiency Moat: Why China Is Beating the U.S. on AI and Everything Else",
    "What could Functional Architecture mean? [video]",
    "Big Shot On The East Coast: The History of the Zoo York Mixtape",
    "rust: Empowering everyone to build reliable and efficient software.",
    "Bitcoin trader recovers wallet with help of Claude",
    "Explore every PPP loan on an interactive map",
    "tensorflow: An Open Source Machine Learning Framework for Everyone",
    "Mullvad exit IPs are surprisingly identifying",
    "Dmitry Senin - I escaped Vladimir Putin in the belly of a dead cow",
    "Why should a Trace-ID be 128 bits?",
    "US plans to indict Cuba's Raul Castro, US DOJ official says",
    "Heads up: new Google support scam uses a REAL email from Google: sysadmin",
    "AI Wellbeing – Measuring and Improving the Functional Pleasure and Pain of AIs",
    "Show HN: Trailmaps.app – Mobile maps that match the trail",
    "Systematically Auditing AI Agent Benchmarks with BenchJack",
    "Countdown to Apophis Close Approach–Cascading Hazards from Asteroid Impacts",
    "I ran forensics on closed models and discovered no one is using dense attention",
    "Kickstarter is forced to ban adult content by payment processors",
    "Latvian government collapses after Ukrainian drones strike oil facility",
    "PSVL 1.0 – The most comprehensive source-visible license (276 clauses)",
    "A message from President Kornbluth about funding and the talent pipeline",
    "How the Ingredients of Life Make Our Journey Worthwhile",
    "Show HN: Openvid – open-source cinematic screen recorder and mockup editor",
    "Social Media Bans Are for Kids. What About Adults?",
    "Show HN: Parse LLM Markdown streams incrementally on the server or client",
    "Tensions flare near Strait of Hormuz as a ship is seized and another is sunk",
    "Multi-LLM trading harness with live leaderboard on Alpaca paper trades",
    "Pope decries rise of AI-directed warfare, saying it leads to a spiral of annihilation",
    "CIA Director John Ratcliffe met with Raul Castro's grandson in Havana, US and Cuban officials say",
    "The Preparation of Programs for an Electronic Digital Computer",
    "DeepSeek V4: The Open-Source Model Frontier Labs Feared",
    "buffa – zero-copy Protobuf lib for Rust",
    "Whom the Gods Would Destroy, They First Give Real-Time Analytics (2013)",
    "Make It Blink: Over-the-Air Exploitation of the Philips Hue Bridge",
    "AI that improves itself indefinitely",
    "Turn a bare VPS into an operational fortress in 15 minutes and 1 command",
    "Establishing AI and data sovereignty in the age of autonomous systems",
    "Get best online coupons and promo codes and deals of hot brands",
    "Court upholds discrimination ruling after male excluded from female-only app",
    "Wood burning reintroduces lead poisoning in US",
    "Building ML framework with Rust and Category Theory",
    "Where's Ed: Anthropic Told Court $5B but Public $19B",
    "Actions is experiencing degraded availability",
    "The chaotic development of Modern Warfare3 – A Documentary",
    "1/4 of this month's Crypto-Gram articles are about Mythos",
    "Show HN: Using the same method from TikTok to concentrate better on meetings",
    "Which Trump cabinet member has a new reality show? The quiz knows",
    "Death toll in attack on Kyiv apartment building now stands at 24",
    "Apple's iPhone 18 Modem Switch Comes with a Quiet Privacy Benefit",
    "Show HN: My time has come – let Claude Code wrap up before 5-hour usage runs out",
    "Ways in which GenAI has changed the way I write code so far",
    "Ask HN: How do you listen to research papers? (TTS workflows for commutes)",
    "The Earliest Known Dentistry Wasn't Done by Our Species",
    "New arXiv policy: 1-year ban for hallucinated references",
    "Scryve-tools – Unified wallet auth for CKB, EVM, and BTC in one NPM package",
    "RelaxAI – UK sovereign LLM inference at 80% cheaper than OpenAI/Claude",
    "Show HN: OrcaSheets, local first analytics engine to process billions of rows",
    "Tachyons Neo – Utility CSS without build step",
    "Developer Experience Is a Performance Feature",
    "Clippy can send emails, write excel sheets, and interact with any application",
    "Chinese short dramas became AI content machines",
    "Prolog Basics Explained with Pokémon",
    "AWS racks M3 Ultra Macs that boast specs you can't currently buy",
    "Why $207M in AI Spend Hasn't Fixed Corporate Slide Decks",
    "Mercurial, 20 years and counting: how are we still alive and kicking? [video]",
    "SCOTUS upholds abortion pill telehealth access. And, Trump returns from China visit",
    "Feedback on a runtime-agnostic AI agent workflow spec (LangGraph/Mastra)",
    "Refactor: Reduce usage of unsafe across Bun Rust codebase ;)",
    "Meta to Start Capturing Employee Mouse Movements, Keystrokes for AI Training",
    "Big Data Expo North America 2026",
    "Health coverage is getting killed by Google AI Overviews",
    "Bitwarden scrubs 'Always free' and 'Inclusion' values from its site",
    "Omnisearch – A lightweight metasearch engine written in C",
    "Truth, Power, and Honest Journalism",
    "Ask HN: Hacker News is suffocating me",
    "Microsoft to automatically roll back faulty Windows drivers",
    "The Supreme Court just told every freight broker that they can be sued",
    "Big Tech groups launch global borrowing spree to fund AI expansion",
    "Show HN: TongueType – Local, privacy-focused Whisper dictation for macOS",
    "Update on GitHub Copilot usage-based pricing",
    "Codex is now in the ChatGPT mobile app",
    "#1 on the leading AI memory benchmark using a smaller, cheaper model",
    "Neanderthal dentists used stone drills to treat cavities nearly 60k years ago",
    "Lessons Learned Building High-Performance Rust Profiler",
    "The U.S. has 1,200 AI bills and no good test for any of them",
    "Anyone accepted crypto payments from customers?",
    "C++26: Standard Library Hardening",
    "Crypto-Agility Is a Runtime Property, Not a Compliance Checkbox",
    "Recursant, the open source AI control plane, now supports OpenClaw",
    "Microsoft and Apple bets on new mascots in bid to seem more cuddly",
    "Mental bugs due to lack of imagination",
    "GitLab is betting a 19th-century economic theory will shape its AI era",
    "I like it, but I hate it more (AI)",
    "Bun single, statically-linked musl binary",
    "[C语言——数据类型，常量，变量，运算符和关键字原创 - CSDN博客] 文章浏览阅读1.1k次，点赞27次，收藏11次。C语言中，数据类型包括整型、浮点型、字符型等，用于定义变量和常量。变量是可变存储单元，常量值不可更改。运算",
    "[C语言入门：数据类型、变量与运算符详解- ycfenxi - 博客园] C语言是编程世界的基石，理解其数据类型、变量和运算符是掌握这门语言的第一步。本文将以简洁的方式，带你深入剖析这些核心概念，并顺便对比其他语言（",
    "[【C语言】03-数据类型、运算符与表达式 - 知乎专栏] 一、简单赋值运算符 1.符号： = 2.格式： 变量标识符=表达式 3.作用：将一个数据（常量或表达式）赋给一个变量 4.左侧必须是变量，不能是常量或表达式",
    "[c语言控制语句（if, switch, for, while等 - 稀土掘金] 执行顺序： 首先定义循环变量并且赋值，然后判断条件是否满足，如果满足就进入循环执行语句块，以及执行增值或减值语句。然后再进行判断，如果满足再次进入循环",
    "[06、C语言流程控制详解- if/switch/for/while循环完整教程 - 程序厨] if-else 语句用于基于条件执行不同的代码块。 · switch-case 语句用于基于变量的值执行不同的代码块，适用于多个分支的情况。 · for 循环用于在给定范围内",
    "[C语言中简单的控制语句if、switch、for、while、goto 原创 - CSDN博客] 首先执行循环体内的代码。 · 然后检查 while 后面的条件。 · 如果条件为真，再次执行循环体。 · 如果条件为假（false），退出循环。",
    "[嵌套调用链式访问· 递归）_c语言:函数调用、参数传递 - CSDN博客] C语言函数核心详解：定义、调用、参数传递、递归与变量作用域. 本知识点体系全面覆盖C语言函数的完整生命周期与运行机理：从函数的本质定位（程序基本",
    "[8.C语言函数详解 - CSDN博客] C语言函数详解：定义、调用、递归、参数传递与变量作用域。C程序设计第8章系统地阐述了C语言中函数这一核心机制的全部关键知识点，涵盖函数",
    "[C语言函数详解：从定义到实践的完整指南 - 腾讯云] C语言函数是模块化编程的核心，包含定义、调用、参数传递等关键概念。文章详细解析了函数语法结构、返回值类型、参数列表及递归函数等进阶用法，帮助开发者实现代码复",
    "[《深入理解C语言指针》精华摘录与解读（一）初识指针&内存分配 ...] 指针的深入概念和使用时的注意事项以及内存管理分配的各种函数（malloc和free）_c 语言指针malloc 如何 ... 数组的内容可以是字符串，因为字符串",
    "[【C语言指针进阶讲解】第五章：指针与内存管理 - CSDN博客] malloc ：. 用于分配指定大小的内存块，返回指向内存块开头的指针。 返回的内存块未被初始化。 · calloc ：. 用于分配指定数量相同大小的内存块，并初始化为0。",
    "[C语言动态内存管理：malloc与free的完整实践指南_文心快码 - Comate] 本文详细介绍了C语言动态内存管理的核心方法，包括`malloc`分配内存和`free`释放内存的完整流程。通过整数数组和字符串两个完整示例，演示了内存计算、",
    "[【C语言】自定义类型：联合体和枚举- typedef union用法 - CSDN博客] 联合体（Union）和枚举（Enum）是C语言中用于定义自定义数据类型的两种形式。它们提供了不同的功能和用法，允许程序员根据实际需要扩展语言的基本类型系统。",
    "[C语言自定义数据类型详解：结构体、联合体、枚举与类型定义 - comate] typedef 关键字用于为现有的数据类型定义一个新的名字（别名）。定义时，指定现有数据类型和新别名。 ... 1typedef struct { 2 int x, y; 3} Point;. 使用别名时，直接使用新类型",
    "[C语言typedef enum的用法详解（附带示例）] 在C语言中，typedef enum 是一种组合用法。typedef 用来为数据类型起一个别名，enum 用来定义枚举类型，组合起来使用的意思就是：为枚举类型起一个别名。",
    "[【C语言进阶篇】常用动态内存分配malloc calloc realloc free - 华为云] 3️⃣ 动态内存函数calloc · 函数的功能是为num 个大小为size 的元素开辟一块空间，并且把空间的每个字节初始化为0。 · 与函数malloc 的区别只在于calloc 会在",
    "[【C语言】动态内存管理全解析：malloc、calloc、realloc与free的 ...] 动态内存分配技术为程序提供了运行时按需分配内存的能力，极大地增强了程序的灵活性和资源利用率。本文将深入讲解C语言中动态内存分配的四大关键函数：malloc、calloc、",
    "[malloc、calloc、realloc、free内存动态函数使用 - 稀土掘金] 本节主要对内存分配问题进行介绍。 动态管理包括malloc、calloc、realloc和free4个函数，其中free函数是用来释放内存空间的。",
    "[单向链表栈二叉树的c语言实现原创 - CSDN博客] 本篇将详细介绍C语言中单向链表的12个基本操作，这些操作对于理解和应用链表至关重要。 1. **创建链表**: 创建链表首先需要定义一个节点结构体，通常包含",
    "[C语言链表栈与队列二叉树遍历 - Dong] 1 链表在物理空间的存储不是连续的，全是操作指针。但也是线性结构。 · 2 分单向链表和双向链表。 · 3 链表节点一般是一个结构体，每个链表节点有自己的数据块",
    "[C语言实现基础数据结构：链表、栈与队列详解 - CSDN博客] 链表的优势在于可以动态地分配内存，不需要预先知道数据量的大小，插入和删除操作的时间复杂度为O(1)（在已知位置的情况下）。",
    "[C语言位运算（按位与运算、或运算、异或运算、左移运算] 按位与运算通常用来对某些位清0，或者保留某些位。例如要把n 的高16 位清0 ，保留低16 位，可以进行 n & 0XFFFF 运算（0XFFFF 在内存中",
    "[C语言位运算符：与、或、异或、取反、左移和右移！] 语言位运算符：与、或、异或、取反、左移和右移位运算是指按二进制进行的运算。在系统软件中，常常需要处理二进制位的问题。C语言提供了6个位操作运算符",
    "c语言中存在6个位操作运算符，且它们只能用于整形操作数。 & 按位与. | 按位或. ^ 按位异或. << 按位左移. >> 按位右移. ~ 按位取反. 1..按位与（AND）：&.",
    "[c语言入门27，多文件编程，extern和static关键字 - 刘冲的博客] c语言入门26，宏定义的使用 · c语言入门27，多文件编程，extern和static关键字 · c语言入门28，头文件的使用 · c语言实战29，linux 内核是如何创建进程，线程的？",
    "[C语言中extern和头文件以及静态动态库概念梳理 - 烛影小札] 通过头文件来调用库功能。在很多场合，源代码不便（或不准）向用户公布，只要向用户提供头文件和二进制的库即可。 · 多文件编译。将稍大的项目分成几个文件实现",
    "[extern的使用详解（多文件编程）——C语言- Luv3 - 博客园] extern是C语言中的一个关键字，一般用在变量名前或函数名前，作用是用来说明“此变量/函数是在别处定义的，要在此处引用”，extern这个关键字大部分读者应该",
    "[【C语言】解决C语言报错：Segmentation Fault - 腾讯云] Buffer Overflow（缓冲区溢出）是C语言中常见且严重的内存管理错误之一。它通常在程序试图写入数据到缓冲区时，超过了缓冲区的边界，覆盖了相邻内存区域。这种错误会",
    "[C语言新手必踩的10大坑：段错误、野指针与缓冲区溢出全解析] 缓冲区溢出是指程序在向缓冲区写入数据时，写入的数据量超过了缓冲区的容量，导致数据覆盖了相邻的内存区域。缓冲区溢出可能会导致程序崩溃、数据泄露甚至被",
    "[C、C++常见内存操作问题内存溢出内存泄漏内存越界缓冲区溢出原创] 1、内存溢出out of memory. 内存溢出out of memory，是指程序在申请内存时，没有足够的内存空间供其使用，出现out of memory。 如malloc new等操作。",
    "[C语言的Socket编程例子（TCP和UDP）_c语言tcp - CSDN博客] 总结起来，C语言中的TCP和UDP Socket编程涉及到创建套接字、设置地址结构体、绑定、监听（TCP独有）、接收连接（TCP独有）、多线程处理（TCP示例中用于并发处理）",
    "#include  #include  #include  #include  #include  #include  #define PORT 8080 #define MAXLINE 1024 int main() { int sockfd, connfd; struct sockaddr_in servaddr, cliaddr; socklen_t len = sizeof(cliaddr",
    "[Windows下C语言的Socket编程例子（TCP和UDP）-阿里云开发者社区] 本文档详细介绍了使用UDP 协议进行通信的过程，包括创建套接字、发送与接收消息等关键步骤。首先，通过`socket()` 函数创建套接字，并设置相应的参数。接着，使用`sendto()`",
    "[关于信息论中熵、相对熵、条件熵、互信息、典型集的一些思考 - 郑瀚 - 博客园] # 关于信息论中熵、相对熵、条件熵、互信息、典型集的一些思考. ## 0x1：信息论与其他学科之间的关系. 香农（shannon）证明了只要通信速率低于信道容量，总可以使误差概率接近于零。同时，香农还进一步讨论了诸如音乐和语音等随机信号都有一个不可再降低的复杂度。遵从热力学的习惯，他将这个临界复杂度命名为熵，并且讨",
    "[信息论基础：从香农熵到互信息的核心概念解析 - CSDN博客] 信息论基础：香农熵、信道容量与编码原理入门教程. 本课程所涵盖的九大标签绝非孤立概念，而是一张环环相扣、层层递进的知识网络：从熵出发定义信息，经互信息",
    "[信息理论与编码（第2版）_百度百科] 《信息理论与编码（第2版）》由吕锋、王虹、刘皓春编著，人民邮电出版社于2013年1月出版，是面向高等院校通信与信息类专业的理论教材。该书以香农信息论为核心框架，结合现代通信技术需求，系统整合信息处理与编码技术的理论与实践内容 [1]。. 1.1.1　信息概念的复杂性　1. 1.1.2　信息的定义　3. 2.3.1　自信息量　14. 2.3.4　自信息量的性质",
    "[LZ77与LZ78 - 维基百科，自由的百科全书] LZ77最初是带有“滑动窗”（Slide window）的压缩算法，这ω个算法后来证明等同于LZ78中首次出现的显式字典编码技术。",
    "[LZ编码技术解析：从基础原理到高效实现 - 百度智能云] 在数据压缩领域，LZ编码（Lempel-Ziv Coding）作为无损压缩的经典算法，其变种LZ77和LZ78被广泛应用于各类文件格式（如PNG、ZIP）和存储系统中。",
    "[数据压缩算法有哪些?数据压缩原理是什么 - 快快网络] ‌霍夫曼编码‌：基于字符频率分配变长编码，常用于JPEG、MP3等格式。 ‌DEFLATE‌：结合LZ77与霍夫曼编码，为ZIP文件的核心算法",
    "[信道编码合集：Turbo码、LDPC码和卷积码 - CSDN博客] Turbo码的译码采用迭代译码算法，通常基于BCJR算法或软输出维特比算法（SOVA）。通过在两个译码器之间传递软信息，逐步逼近正确的译码结果。",
    "[信道编码与信源编码基本 - CSDN博客] 在信源编码阶段，LDPC编码用于对原始无损数据进行处理，目的是在不改变数据含义的前提下，降低数据的传输或存储需求。通过LDPC编码，可以将数据压缩成更短的码",
    "[信道编码_百度百科] # 信道编码. ## 发展简史. 人类在信道编码上的第一次突破发生在1949年。R.Hamming和M.Golay提出了第一个实用的差错控制编码方案——汉明码。. Golay码之后是一种的新的分组码——RM码。在1969年到1977年之间，RM码广泛应用于火星探测，同时，其快速的译码算法非常适合于光纤通信系统。. RM码之后人们又提出了循环码的概念，也叫循环冗余校验(CRC",
    "[密码发展史之近现代密码_国家密码管理局门户] # 密码发展史之近现代密码. 电报的出现第一次使远距离快速传递信息成为可能,事实上,它增强了西方各国的通讯能力;20世纪初,意大利物理学家奎里亚摩•马可尼发明了无线电报,让无线电波成为新的通讯手段,它实现了远距离通讯的即时传输,但是通过无线电波送出的每条信息不仅传给了己方,也传送给了敌方,因此这就意味着必须给每条信息加密,随着第一次世界大战的爆发,对",
    "[椭圆曲线密码学简介（三）：ECDH加密算法和ECDSA数字签名算法] 这是一个用于产生方程系数和/或基准点的随机数。这些参数通过计算种子的哈希获得。众所周知，哈希的计算是“easy”的，但是求逆是“hard”的",
    "[密码学简介 - CTF Wiki] # 密码学简介¶. The current page still doesn't have a translation for this language. You can read it through Google Translate. Besides, you can also help to translate it: Contributing. 其中",
    "[[算法] OFDM信号的调制与解调详解（完整仿真代码） 原创 - CSDN博客] OFDM（正交频分复用）的核心思想是将高速数据流分割为多个低速子流，通过相互正交的子载波并行传输。其数学基础是离散傅里叶变换（DFT） 和奈奎斯特采样定理。",
    "[采样定理与傅里叶变换 - 知乎专栏] 它阐明了一个重要的原则：为了能够无误地从离散采样中重构原始的连续信号，所需的采样频率（即采样率）至少应是信号最高频率成分的两倍。 重要的是要区分信号的频率与采样率这",
    "[浅谈几个通信概念-如何理解卷积，负频率，傅里叶变换 - CSDN博客] 数字信号处理核心原理与关键技术：傅里叶变换、卷积、采样定理及正交函数分析. 奈奎斯特采样定理（Nyquist-Shannon Sampling Theorem）是连接模拟与",
    "[一种应用于5G基于LDPC码的物理层包编码] packet code; physical layer; single parity code; min-sum algorithm; iterative decode; low density parity check codes 中图分类号： TN929.5 文献标志码： A 文章编号： 1009-6868 (2016) 03-0026-005 ",
    "[[论文评述] Demystifying 5G Polar and LDPC Codes] 低密度奇偶校验（LDPC）编码是线性区块编码的一种，它基于稀疏奇偶校验矩阵，并利用图算法进行迭代解码。LDPC编码在5G NR中被认为是理想的候选方案，原因在于其出色的纠错能力、",
    "[适用于5G 系统的低密度奇偶检查（LDPC）代码Intel® FPGA IP] 随着无线供货商开始部署基于5G 的技术，低密度奇偶校验（LDPC） 代码正在取代涡轮代码，成为前向纠错的首选编码。 本训练介绍了LDPC 码IP 核Intel® FPGA，即编码器和译码器区块",
    "[数字签名算法RSA - 阿里云开发者社区] RSA是一种基于数论的非对称加密算法，依赖大整数质因数分解的困难性保证安全性。它生成公钥和私钥，公钥加密，私钥解密，适用于数据加密、数字签名和互联网安全等领域。",
    "[RSA密钥、加密与数字签名：原理与实践 - 百度智能云] 与传统的对称加密算法（如DES）相比，RSA算法使用两个密钥：公钥和私钥。公钥用于加密数据和验证数字签名，而私钥用于解密数据和创建数字签名。这种密钥分发的",
    "[RSA加密原理与RSA公钥加密系统、数字签名原创 - CSDN博客] 从加密过程来讲，公钥加密使用的接受方的公钥和密钥，而数字签名使用的发送方的公钥和密钥。 公钥加密是发送者运用信息接收方的公钥进行加密，然后接受者用",
    "[DCT变换和JPEG原理 - 知乎专栏] DCT并不能压缩数据，但下一步量化可以。 4 量化. 这一步又存在一个量化表。 将上一步得到的常量次数表除以量化表（这个量化表通常趋势是左上角小，右",
    "[JPEG编码中的DCT与量化_jpeg反向量化 - CSDN博客] 首先，理解JPEG编码的核心是离散余弦变换（Discrete Cosine Transform, DCT）。DCT将图像数据从空间域转换到频率域，使得高频信息更容易被压缩，而人眼对高频",
    "[JPEG图像压缩算法流程详解- 梁君牧- 博客园] 图像数据转换为DCT频率系数之后，还要进行量化阶段，才能进入编码过程。量化阶段需要两个8x8量化矩阵数据，一个是专门处理亮度的频率系数，另一个则是针对色度",
    "[完整教程：正交频分复用技术- yangykaifa - 博客园] 正交频分复用（OFDM）通过将宽带信道划分为大量正交的窄带子信道，并巧妙 ... 这正是OFDM成为现代宽带无线通信（如Wi-Fi， 4G/5G， DVB-T等）基石技术的原因。",
    "[OFDM技术应用及其发展趋势] # OFDM技术应用及其发展趋势. *   Vol. 10 No. 5 (October 2020). #### 期刊菜单. OFDM Technology and Its Development Trend. **DOI:**10.12677/HJWC.2020.105008, PDF, 被引量. **作者:**吕浩田, 张 涛, 苏发富：69224部队，新疆 ",
    "| |  | 调幅 | 调频 | 调角 | | --- | --- | --- | --- | | **模拟** | AM （SSB · DSB） | FM | PM | | **数字** | ASK （OOK · QAM） | FSK （MSK · GFSK） | PSK （CPM） | | **其他** | SM（英语：Space modulation） （模拟） | | |. | **其他*",
    "[[PDF] FFT (快速傅里叶变换) 波形分析] 康泰电子 010-62329030 010-62329880 010-62329881 网址：www.quatronix-cn.com FFT (快速傅里叶变换) 波形分析 Calculate FFTs with WinDaq Waveform Browser Available for free with every DATAQ Instru",
    "[FFT是什么？快速傅里叶变换(FFT变换)原理 - 知乎专栏] 离散傅里叶变换(DFT) 算法将信号样本从时域变换到频域。DFT 广泛应用于频谱分析、应用力学、声学、医学成像、数值分析、仪器仪表和电信领域。 对具有N个",
    "[FFT快速傅里叶变换基础FFT basic - 知乎专栏] 傅里叶分析是信号处理的重要理论基础，简单来说，傅里叶变换可以将一个时域信号转换为一个对应的频域表达；反之，若已知信号的频率响应，则可通过傅里叶逆变换",
    "[密码学：一文读懂对称密钥体系 - 知乎专栏] AES（Advanced Encryption Standard）是一种对称分组密码标准，广泛应用于保护敏感数据的加密通信中。它是DES的后继者，经过了广泛的评估和认可，成为当前最常用的对称加密算法",
    "[对称加密算法和分组密码的模式 - 腾讯云] DES、3DES、AES等大多数对称密码算法都属于分组密码。 ECB模式. 全称Electronic CodeBook mode，电子密码本模式。 分组方式：将明文分组加密之后的结果直接称为密文分组。",
    "[密码学——AES/DES加密算法原理介绍 - 枫のBlog] Crypto 3CTF 19Docker 6Java 13Misc 1Web 13学习 18安全 8未分类 3漏洞复现 11隐私计算 1. Crypto 4CTF 16Docker 6Java 13Misc 1Web 11学习 10安全 7漏洞复现 9隐私计算 1. AES（Advanced Encryption Standard），",
    "如果主问题为凸优化问题，目标函数是凸函数并且可行域是凸集，且存在一点在不等式约束下严格成立，那么强对偶性通常成立。讲解中提到了常见的优化策略，如利用KKT条件来求解有",
    "由于对偶函数是凹函数，故拉格朗日对偶问题一定是凸优化问题，其对应的最优解为\lambda^*,v^* （最优拉格朗日乘子），若对应的最优值为d^* ，则总有d^*\leq p* 。 当d",
    "这些求解条件就是KKT条件。(1)是对拉格朗日函数取极值时候带来的一个必要条件，(2)是拉格朗日系数约束（同等式情况），(3)是不等式约束情况，(4)是互补松弛条件，(5)",
    "梯度下降法是一种迭代算法，选取适当的初值x(0) x ( 0 ) ，不断迭代，更新x x 的值，进行目标函数的极小化，直至收敛。由于负梯度方向是使函数值下降最快的方向",
    "3 BFGS 牛顿法&拟牛顿法设无约束优化问题： min⁡f(x), x∈Rn\min f(x) ... Matlab支持多种优化算法，包括梯度下降法、牛顿法、拟牛顿法（如BFGS和",
    "本文讲解的是无约束优化中几个常见的基于梯度的方法，主要有梯度下降与牛顿方法、BFGS 与L-BFGS 算法。 梯度下降法是基于目标函数梯度的，算法的收敛",
    "Title: 约束优化：拉格朗日乘子法与罚函数法比较,-CSDN博客",
    "L(x,y,λ)=f(x,y)+λ⋅g(x,y) L(x, y, \lambda) = f(x, y) + \lambda \cdot g(x, y)L(x,y,λ)=f(x,y)+λ⋅g(x,y). L(x,y,λ)=f(x,y)+λ⋅g(x,y)L(x,y,λ)=x2+y2+λ⋅(x+y−3) L(x, y, \lambda) = f(x, y) + \lambda \cdot g(x, y)",
    "# 增广拉格朗日函数(The augmented Lagrangian)及其KKT条件. 最新推荐文章于 2024-12-20 09:28:24 发布. 于 2020-09-04 10:36:12 发布. CC 4.0 BY-SA版权. 版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。. Ψ(x,λ,ν)=L(x,λ,ν)+α2∑j=1m(λ",
    "最新推荐文章于 2023-02-21 22:35:07 发布. 于 2019-01-07 17:13:56 发布. CC 4.0 BY-SA版权. ### 增广拉格朗日函数法（ Augmented Lagrangian method）. #### 一、等式约束. minxf(x)(x)=0,i=1,⋯,m. minxs.t.f(x)ci(x)=0,i=1,⋯,m. minx​s.t.​f(x)ci",
    "1 对偶单纯型法. 在上一节笔记中我们研究了线性规划问题的对偶问题，并且我们根据强对偶定理可知，线性规划问题的原问题和对偶问题是等价的。这就自然而然让我们产生一个",
    "第2章线性规划及单纯形法 2.1线性规划问题的提出与数学模型 2.2两个变量的图解法 2.3线性规划的标准形与各种解 2.4单纯形法原理 2.5单纯形法 2.6单纯形法的进一步讨论",
    "在包含该闭区域的更大的区域上解析。解析函数也称全纯函数。 如果f在z_0的任何邻域都不解析",
    "第一篇 复变函数 复数的概念起源于求方程的根，在二次、三次代数方程的求根中就出现了负 数开平方的情况．在很长时间里，人们对这类数不能理解．但随着数学的发展， 这类数的重要性就日益显现出来． 复数的一般形式是： i a b + ， 其中i 是虚数单位． 以复数作为自变量的函数叫做复变函数，而与之相关的理论就是复变函数 论．解析函数是复变函数中一类具有解析性质的函数，复变函数论主要研究复数 域上的解析",
    "在现代数学中，全纯函数的概念已经被推广到多复变函数和高维复流形上。这些推广不仅丰富了理论本身，也为解决更复杂的数学和物理问题提供了新的工具。 在应用数学领域",
    "莫雷拉定理. 柯西積分公式. 柯西留數定理. 幅角原理. 魯歇定理. 卡索拉蒂-魏爾斯特拉斯定理. 留數. 洛朗級數 ... 餘弦函數是非常數函數，但它有泰勒級數展開",
    "... 泰勒展开，即没有负幂项的洛朗展开；. 对于极点，函数在其附近的洛朗展开只有有限个负幂项；. 于是，对于孤立奇点，函数在其附近的洛朗展开必然有无限多个负幂项",
    "在複分析中，留數定理，又叫殘數定理（英語：Residue theorem），是用來計算解析函數沿著閉曲線的路徑積分的一個有力的工具，也可以用來計算實函數的積分。它是柯西積分定理和",
    "在解析变换中，研究复变函数的映射性质的重要主题是共形映射。如果一个复变函数f ( z ) {\displaystyle f(z)} 在区域D {\displaystyle D} 内是一个单射且保角的变换，",
    "本文，我们从几何的观点来研究全纯函数，也就是将全纯函数作为平面上的映射来讨论。 自从26年初公式编辑器改版后，手机端的公式显示就变得十分差劲，推荐大家",
    "· 118 · 第三章 解析延拓 图55 元F0 = (U0, f0) 为在第27 小节中考察过的例子, 其中U0 = {|z −1| < 1}, 而f0 为满足条件−π/2 < arg z < π/2 的√z 的分支, 它沿着任意弧γs, s ̸= 1/2 (如上面所 说, 对于这样的延拓只要arg z 沿该弧连续变化即可). 函数(9) 几何 地将D0 映射成带状区域D∗ 0 = {w : −",
    "友好的 实分析导读 II 引言 我决定在实分析的 II 篇继续写实变函数论的部分。之前的 I 篇大多都着重在测度论，那现在是时候积 分迈进了。估计之后的实分导读系列会写抽象空间方面的东西，“导读”类文我一般会写比较广的概 念。 前置知识 黎曼积分 （据我所问）高中或以上的读者应该已经接触过定积分的概念了，一般教材里提到的定积分就是指黎 曼积分。对于定义在有限闭区间上的函数，黎曼积分提供了一种计算曲",
    "在测度论中，**勒贝格测度**（Lebesgue measure）是欧几里得空间上的标准测度。对维数为1，2，3的情况，勒贝格测度就是通常的长度、面积、体积。它广泛应用于实分析，特别是用于定义勒贝格积分。可以赋予勒贝格测度的集合称为**勒贝格可测集**；勒贝格可测集 *A* 的测度记作 *λ* (*A*) 。一般來說，我們允許一个集合的勒贝格测度为 ∞ ，但是即使如此，在假设选择公理成立时，**R",
    "测度是定义在某个集合的σ-代数（可测集类）上的函数，满足：. - 非负 ... 可测函数、一般测度空间. 收敛性处理, 需要一致收敛, 通过MCT/DCT处理逐点",
    "武汉大学泛函分析精品课程教材：希尔伯特空间、巴拿赫空间与算子理论. 当该空间关于此范数完备时，即为巴拿赫空间——这是泛函分析最基础也最普适的舞台",
    "巴拿赫空间中最重要的特例被称为希尔伯特空间，其上的范数由一个内积导出。这类空间是量子力学数学描述的基础。更一般的泛函分析也研究Fréchet空间和拓扑向量空间等没有定义",
    "巴拿赫的《线性算子理论》（1932 年）系统总结了赋范空间理论，成为泛函分析的“圣经”。 四、索伯列夫空间（1930-1940）：偏微分方程的弱解革命. 提出背景：苏联人",
    "《泛函分析》教学大纲 课程编码：1511102302 课程名称：泛函分析 学时/学分：32/2 先修课程： 《数学分析》 、 《实变函数》 适用专业：数学与应用数学 开课教研室：分析与方程教研室 一、课程性质与任务 1．课程性质：本课程是数学与应用数学专业的一门专业选修课，是现代数学中的一个 较新的重要分支，它综合地运用分析、代数和几何的观点与方法，研究分析数学，现代物理 和现代工程技术提出的许多",
    "第三章 Ｈｉｌｂｅｒｔ空间上的 有界线性算子 由于Ｈｉｌｂｅｒｔ空间中存在正交基，因此其空间的几何性质比Ｂａｎａｃｈ空间要 丰富得多，Ｈｉｌｂｅｒｔ空间上的有界线性算子理论也比Ｂａｎａｃｈ空间情形更加深刻， 特别是Ｈｉｌｂｅｒｔ空间上的自伴算子理论形成了有界线性算子理论中的最完美部分． § １ 投影定理与ＦｒéｃｈｅｔＲｉｅｓｚ表示定理 １．１ 投影定理 我们在第二章中曾指出，Ｂａｎａｃｈ空间的",
    "章节中介绍了Hilbert 空间上有界线性泛函的Riesz 表示定理，以及四个重要的线性算子理论定理。还定义了有界线性算子和有 ... 2由此也可说明有穷维线性空间上的线性算子必有界.",
    "通过变分法，哈密顿原理可以导出拉格朗日方程：. 对作用量 S 进行变分 ... 哈密顿原理是经典力学的核心，通过作用量的极值条件描述系统的运动规律，具有深刻的",
    "哈密顿原理是经典力学中的一个基本原理，它在变分法中占有核心地位，能够用来推导所有力学定律。这个原理由威廉·哈密顿提出，为理解和处理物理系统的运动提供",
    "哈密顿原理可表述对一个系统的函数进行积分，当该积分的变分为0 的 ... 但是，欧拉-拉格朗日方程其实是变分法得出的结果，也是最小作用量原理的",
    "信息论基础理论：熵、互信息与信道容量核心概念解析. 信道容量C=max ... 熵、条件熵、互信息、信道容量和率失真等。文件的具体组织结构和使用方式",
    ", x1) = n X i=1 H (Xi | Xi−1, · · · , X1) 这里所体现的对于联合熵的不同展开方式，在解题中常常会用到！ 16 趁现在还有期待 信息论学习指导 高源 3.1.7 相对熵 D(p∥q) = X x∈X p(x) log p(x) q(x) = Ep log p(X) q(X) 3.1.8 相对熵的性质 1. 这 样就可以确定，输入服从均匀分布时，熵H(Y ) 取",
    "连续信道的容量. 模拟信道容量. 信道编码. Lecture_3. 6 第四章信息速率失真函数与熵压缩编码. 问题的提出. 率失真理论的基本概念. 率失真函数的性质. 率失真函数的计算.",
    "*   [创建账号](https://zh.wikipedia.org/w/index.php?title=Special:CreateAccount&returnto=%E6%9D%8E%E9%9B%85%E6%99%AE%E8%AF%BA%E5%A4%AB%E7%A8%B3%E5%AE%9A%E6%80%A7&returntoquery=variant%3Dzh-cn "我们推荐您创建账号并登",
    "Calculus of Variations and Optimal Control. Discrete-Time Optimal Control Systems. The discrete-time Kalman filter. The continuous-time Kalman filter. 关于稳定的一些提法，最核心的是：李雅普诺夫稳定性、渐进稳定性、大范围渐进稳定。还有像极限定义一样的",
    "现代控制理论核心教材：状态空间法、能控性与能观性、最优控制及李雅普诺夫稳定性分析. 李雅普诺夫稳定性理论是现代控制中分析非线性与线性系统稳定性",
    "文章目录1 、优化模型1.1 数学规划模型1.2 微分方程组模型1.3 图论与网络优化问题1.4 概率模型1.5 组合优化经典问题现代优化算法：禁忌搜索；模拟退火；遗传算法；",
    "本文详细介绍了数学建模领域中的十大经典模型，包括线性规划、动态规划、回归分析、微分方程、图论、概率统计、最优化、时间序列分析、神经网络以及模糊数学等。这些模型在",
    "· 随机性模型：明确考虑系统中的随机因素（噪声、不确定性），模型输出是概率分布。例子：包含随机项的微分方程（随机微分方程）、蒙特卡洛模拟、排队论模型、贝",
    "x x y y 2 sin tan = + ′ , 1 ) 0 ( = y Sol. x x f tan ) ( = , x x r 2 sin ) ( = , ∫ ∫ = = = x xdx dx x f x h sec ln tan ) ( ) ( x e e x x h sec sec ln ) ( = = , x x e e e x x x h cos sec 1 sec ln sec l",
    "这里只含有单个自变量，也称为常微分方程。 方程中出现的未知函数的最高阶 ... 这样，Bernoulli方程就化为如下的一阶线性微分方程. \[\dfrac{du}{dx}+(1-n)p(x)u",
    "值的完整积分曲线  ：. DSolve::bvnul 消息指示通解的一个分支（上一个图中的下部分支）没有给出满足给定初始条件 y[1]3 的解：. 以下是线性一阶 ODE，因为其中的  和  都是 1 次，并且  是最高阶导数. 给定的 ODE 在  中可能不是线性的，但可以被视为  中的线性 ODE.",
    "3.3 受迫振动方程的解. 根据方程(3) ，结合阻尼运动的计算式，我们可以发现受迫运动的解就是在之前阻尼运动的通解之上再叠加上自身的特解。 我们当然可以使用2.4中的",
    "# 二阶常系数线性微分方程. 二阶常系数线性微分方程（linear differential equation with constant coefficients of the second order）是形如y''+py'+qy=f(x)的微分方程，其中p，q是实常数。自由项f(x)为定义在区间I上的连续函数，即y''+py'+qy=0时，称为二阶常系数齐次线性微分方程。若函数y1和y2之比为",
    "2 4 0 x y xy y cc c   , 2 1 y x 10. 2 0 x y xy y cc c   , y (1) = 1; (1) 2 yc , x, x ln (x) 19. 2 2 0 y y y cc c   , y (0) = 0, (0) 15 yc , cos x e x  , sin x e x  20.",
    "# 微分方程 Part 5 偏微分方程式 變數分離、經典方程式. 在這系列文章中，我們已經探討了各種微分方程式的類型和解法。這篇文章將進入更為複雜的領域：**偏微分方程式（Partial Differential Equations，簡稱PDEs）**。. 偏微分方程式涉及**多個自變數（independent variables** 的方程式，這與我們之前討論的常微分方程式（ODEs）有所不同。",
    "热传导：在稳态热传导问题中，温度分布满足拉普拉斯方程。 拉普拉斯方程的解称为调和函数，具有许多重要性质，如平均值定理和最大值原理。",
    "4.波动方程： \frac{\partial ^{2}u}{\partial t^{2}}=\Delta u+f ,在声光水学的波动中经常会用到。 5.扩散方程（热传导方程）： \frac{\partial u}{\partial t}=\frac",
    "数学物理方程核心方法与应用：傅里叶变换、格林函数及分离变量法详解. 课件中重点阐述的**分离变量法**，是求解具有规则几何边界（如矩形、圆柱、球",
    "分离变量法又称Fourier方法，而在波动方程情形也称为. 驻波法。它是解决数学物理方程定解问题中的一中基本. 方法，这个方法建立在叠加原理的基础上，其基本出发.",
    "用特征函数是cos(npi x)*cos(mpi y) 的二维傅里叶级数特征函数展开方法，配合系数匹配应该可以。这个函数对应的特征值是pi^2(n^2+m^2)。这很可能导致",
    "数学（下） • 例求 lim 𝑥→+∞ (ln 𝑥)𝑚 𝑥𝑛 , 其中𝑚, 𝑛> 0. 数学（下） • 例求lim 𝑥→1 𝑥 𝑥−1 − 1 ln 𝑥. 数学（下） • 例求 lim 𝑥→+∞ 6 𝑥6 + 𝑥5 − 6 𝑥6 −𝑥5 . 数学（下） 4.3 泰勒中值定理 • 在利用函数的微分作近似计算的时候, 我们有一阶近似公式 𝑓𝑥= 𝑓𝑥0 + 𝑓′ 𝑥0 𝑥−𝑥0 + 𝑜𝑥−𝑥0 . 数",
    "标题为“高等数学上册知识点.pdf”的文件，涉及高等数学上册的相关知识点，主要包含函数与极限、导数与微分、以及微分中值定理与导数的应用等重要章节内容。",
    "往期知识点-数学概念篇. 列1. 1.映射 · 4.函数极限性质 · 7.极限存在准则 · 10.微分中值定理 · 13.曲率 · 16.分布积分法 · 19.无界函数审敛法 · 22.平面",
    "我们已知不定积分是求导的逆运算，而定积分是函数曲线与x 轴之间的面积，二者乍看起来没什么联系，但牛顿—莱布尼兹公式却揭示了了二者之间的重要关系． 若F(x) 是f(x) 的一个原",
    "# 牛顿-莱布尼茨公式. 本词条由中国科学院大学本科部、中国科学院自然科学史研究所 参与编辑并审核，经科普中国·科学百科认证 。. 牛顿-莱布尼兹公式，又称微积分基本定理，最初指微积分理论中由牛顿和莱布尼兹两人分别独立发现的求定积分的方法。该公式将函数的定积分与原函数联系起来，为计算提供了简便的方法。这不仅是一个用于计算的重要的积分公式，也是分析学的基本公式。. 牛顿-莱布尼兹公式可以拓展到高维空",
    "这即为牛顿—莱布尼茨公式。 牛顿-莱布尼茨公式的意义就在于把不定积分与定积分联系了起来，也让定积分的运算有了一个完善、令人满意的方法。 微积分",
    "本篇文章，探讨下多元函数微分学下的一些知识点之间的关系。包括全微分、偏导数、方向导数、梯度、全导数等内容。 初学这些知识的时候，学生会明显觉得",
    "多元函数没有简单的“导数”的概念。但为了研究多元函数在某点的变化率，我们可以考虑“方向导数”。 ... ，取其中的一个方向l=(u0,v0)，并假设该方向与x轴正方向夹",
    "## 向量微积分一文速通:从曲线积分到曲面积分. 文章很短也很长，两类积分，在曲线和曲面上，四种类型，分别在标量和矢量情况下。三个定理，格林，高斯，斯托克斯。以及完整的微积分基本定理，在低维和高维之间互相连通。不管怎么说，会不会算，那是计算问题，如果不会区分，那是真没学会。. 直线积分的自变量取值范围是直线，也就是实数轴，用 x 就可以指明自变量。而曲线积分的自变量取值范围是曲线，需要(x,y)才",
    "同济版高等数学下册核心知识点复习提纲：向量代数、多元微积分与曲线曲面积分. 格林公式架起平面闭合曲线积分与二重积分之间的桥梁，本质是二维斯托克斯公式",
    "斯托克斯定理： 格林公式是斯托克斯定理在二维空间的特例。斯托克斯定理建立了曲面积分与曲线积分之间的关系。 例题就不放了，自己算。 我最喜欢的",
    "# 吴紫航. # 【知识总结】 第七章-无穷级数. 发表于     更新于     分类于  微积分      阅读次数：     评论数：. * 本知识总结不提供完整的理论体系汇总，旨在给出*概念的理解*以及*各类问题的思考框架*。. ## 理解铺垫. + 无穷级数本质是**一个**数，这个数的表达形式是**无穷个**数的和（加号连接每一项）. + 数列本质是**无穷个**数，这无穷个数的表达形",
    "傅里叶级数：任何周期函数都可以用正弦函数和余弦函数构成的无穷级数来表示，称为傅里叶级数。 ... 若幂级数的收敛半径r=\left\{ \begin{matrix} \frac{1}{l} \quad 0<l",
    "两个标杆——等比级数，P级数; 正项级数比较审敛法; 正项级数比值、根植审敛法; 正项级数积分审敛法; 交错级数Leibniz审敛法; 绝对收敛&条件收敛; 绝对收敛&条件收敛的三",
    "... 正规子群与商群，正规化子与中心化子. 第四章下群的构造（下）：自同构，直积与半 ... 群（同态将一个子群映射成另一个子群）; \varphi 的kernel（核",
    "再補充一點，正規子群是群同態的核，這表示：f(gNg⁻¹) = f(g) 1 f(g⁻¹) ... 在四元數群裡，對+/-1 取商會得到一個群，它跟任何子群都不是同構的。",
    "正规子群具有特殊意义，利用关于正规子群的陪集集合，可以构造出新的群，称为商群。当群(G) 的子群(H) 是正规子群时，(G) 中(H) 的左（或右）陪集集合本身构成",
    "多項式環K[x]是最重要的例子之一，既是歐幾里得整环，也是PID和UFD。 分圓域(Cyclotomic field) - 由一個原始n次單位根生成的數域。其整數環不一定是UFD，研究其理想",
    "【环论开始】环; 子环环同态; 【环的重要子结构】理想商环; 【重要定理】第一环同构定理; 整环主理想整环唯一分解整环; UFD和PID的性质; gcd lcm PID→UFD→整环 ... 】多项式",
    "我们用PID表示principal ideal domain，用UFD 表示unique factorization domain. 所有的环都默认为带单位元的交换环。",
    "该域扩张的伽罗瓦群是分裂域上所有保持基域元素不动的自同构之集合。伽罗瓦理论的主定理断言：伽罗瓦 ... 代数基本定理告诉我们，代数方程一定在复数域上有解",
    "商环与扩张域; 同构与自同构; 分裂域与重根; 域扩张次数与自同构的固定域; 域的自同构群; 伽罗瓦理论的基本定理; 有限域; 整系数多项式; 尺规作图的构造; 用根式解方程; 域",
    "伽罗瓦理论基本定理是抽象代数中的定理，通过群的概念来描述特定域扩张的细致结构。定理说明了，如果某个域扩张L/K是有限伽罗瓦扩张，则此扩张的伽罗瓦群的子群与其中间",
    "在这一领域中，如果每一个开集的原像仍是开集，则该映射被称为连续映射，而同胚则是指在空间之间建立拓扑等价关系的连续双射。该学科的核心在于研究连通性、紧致性和维度等",
    "定义2.4 (闭集) ♣ 设F 是拓扑空间(X, τ) 的一个子集，如果F 的补集F c = X\F ∈τ，即补集为开集，则 称F 是一个闭集. 定义2.13 (连续映射(continuous map)) ♣ 如果映射f : X →Y 满足：任取f(x0) 的邻域V ，有f −1(V ) 是x0 的邻域，则称f 在x0 处连 连 连续 续 续. □ 定义2.14 (恒等映射(identity ma",
    "连续性: 开集的原像是开集. 关于与 ... 紧致空间的闭子集是紧的. Hausdorff空间中的紧集是闭集. 紧空间到Hausdorff空间的既单又满的连续映射是同胚.",
    "... ，以区分不同维数欧氏空间为切入点，提出用代数不变量作为拓扑空间的判定工具。先阐释同伦、同伦等价与CW. 148 views · 6 days ago ...more. 北游知. 59.",
    "则由命题3.3.17(5)，f 诱导了基本群之间的映射 f∗: π1(X, x0) →π1(Y, y0), [γ]p 7→[f ◦γ]p. □ 于是对应关系π1 : PointedT OP →GROUP，其中 (X, x0) ⇝π1(X, x0), f ∈C((X, x0), (Y, y0)) ⇝π1(f) = f∗: π1(X, x0) →π1(Y, y0). （基本群是拓扑不变量） ♥ 如果f",
    "我来自物理学背景，学了点微分几何。在我看来，欧拉示性数就是de Rham 上同调向量空间的维度的交替和。我有一些关于它的问题： 为什么这和V-E+F 一样？",
    "我们会通过de Rham复形来构造de Rham上同调。 因为欧几里得空间上 ... Vito Volterra最终在1889年完成了一般的庞加莱引理的证明。 最终人们发现",
    "本文使用Zhihu On VSCode 创作并发布de Rham上同调的Poincaré引理本节我们希望证明, H^*(\mathbb{R}^n\times \mathbb{R}^1)\cong H^*(\mathbb{R}^n).",
    "我们对于这门课的设. 想是介绍微分流形的基本概念和例子，使学生熟悉微分流形上光滑切向量场、外微分式的性质 ... de Rham 上同调，Cech 上同调，关于Laplace",
    "𝜔௡ାଵ𝑥 ሺ2.2.16ሻ 𝑅ଵ𝑥ൌ1 2 𝑓ᇱᇱ𝜉𝜔ଶ𝑥ൌ1 2 𝑓ᇱᇱ𝜉 𝑥െ𝑥଴ 𝑥െ𝑥ଵ，𝜉∈𝑥଴, 𝑥ଵ 𝑅ଶ𝑥ൌ1 6 𝑓ᇱᇱᇱ𝜉 𝑥െ𝑥଴ 𝑥െ𝑥ଵ 𝑥െ𝑥ଶ，𝜉∈𝑥଴, 𝑥ଶ 例2.1 已知 ， ， ，用线性插值 及抛物插值求 ，并估计截断误差 取଴ ，ଵ ，ଶ ，଴ ，ଵ ，ଶ 用线性插值计算，取଴ 及ଵ ，由式 得 sin 0.3367 ൎ𝐿ଵ0.3367 ൌ𝑦଴൅𝑦ଵ",
    "对于y = P(x) ,如果\forall i \in [1, n], P(x_i) = yi ,那么函数P(x) 是数据点(x_1, y_1), \dots,(x_n, y_n) 的插值函数Interpolating function。",
    "拉格朗日插值法：插值多项式和插值基函数的形式对称，容易编程。但是，增加节点时，需要重新计算每一个插值基函数。 牛顿插值法：当插值节点增加时，之前已计算的结果仍然能用，每",
    "数值求积法则是用于近似计算定积分的具体公式，通常将积分表示为被积函数在一组特定点（节点）上的值的加权和。常见的例子包括梯形法则、辛普森法则和高斯求积法。",
    "梯形法则适用于在均匀间隔的采样点处积分来自实验的数据。 这对于表现不佳的函数是有好处的。 辛普森的规则依赖于被积函数的更高阶的近似，以便准确。",
    "所以可以把这种计算用于近似f(x)的积分。辛普森公式是梯形公式的改进形式。另外，我们还可以通过最小二乘法求函数的近似多项式，这种方法称为高斯积分。",
    "线性方程组的数值解法主要包括直接法（如高斯消元法、LU分解）和迭代法（如雅可比迭代法、高斯-赛德尔迭代法）。 高斯消元法的基本步骤是什么？ 高斯消元法的基本步骤包括：1.",
    "求解线性方程组： 高斯消元--LU分解--Jacobi迭代--高斯赛德尔--sor超松弛迭代1.问题概述假定线性方程组 ... 高斯-赛德尔迭代法求解线性方程组数值分析实验.",
    "一般常用直接法为高斯消元法（Gauss elimination）或者是LU 分解（LU decomposition）。 而相对应的，迭代法则是通过有限次的迭代，将数值解不断逼近解析解的过程。因此，迭代法",
    "在其论文中指出热方程的RK4 方法相比于Euler 方法(A.1) 的优点并不明显. 原因如下: 1. RK4 方法的稳定性条件是k/h^2< 0.7, 优越性并不明显;",
    "在之前常微分方程的数值解法系列中，已经介绍了欧拉法，改进欧拉法以及中值法等多种常微分方程的数值解法。但是之前讲解的方法的局部截断误差相对来说比较大",
    "4 G r o n w a l l 不等式 习题 § 2 线性多步法 2 . 3 收敛性和误差估计 习题 § 4 单步法和R u n g e - K u t t a （龙格- 库塔）法 4 . 3 R u n g e - K u t t a 法 习题 § 5 绝对稳定性和绝对稳定域 5 . 3 L O D 法 习题 § 7 数值例子 7 . 3 数值例子 习题 第五章 边值问题的变分形式与R i ",
    "当 m = p m = p m=p 为素数时， φ ( p ) = p − 1 \varphi(p) = p - 1 φ(p)=p−1，欧拉定理退化为费马小定理。因此费马小定理是欧拉定理的特例。",
    "* [简介](https://oi-wiki.org/). * [数学](https://oi-wiki.org/math/). * [数据结构](https://oi-wiki.org/ds/). * [图论](https://oi-wiki.org/graph/). + [数学部分简介](https://oi-wiki.org/math/). + [位操作](https://oi-wiki.o",
    "数学上，当两个整数除以同一个正整数，若得相同余数，则二整数同余。 两个 ... 这就是著名的费马小定理。它是欧拉定理的特例。 欧拉定理是RSA算法的核心。理解了",
    "* [定义](https://oi-wiki.org/math/number-theory/quad-residue/#%E5%AE%9A%E4%B9%89). * [习题](https://oi-wiki.org/math/number-theory/quad-residue/#%E4%B9%A0%E9%A2%98). * [定义](https://oi-wiki.org/math/number",
    "x^{2} \equiv d \quad(\bmod p) ，这里，称d 是模p 的二次剩余。 Legendre 符号就表示了该方程解的三种存在情况。这样看好像单独定义这样的记号显得很麻烦，",
    "... 勒让德在1798年尝试证明二次互反律时引入。勒让德符号主要用于判断一个整数是否是给定奇素数的二次剩余（即能否表示为该素数下某个整数的平方）或二次",
    "# 黎曼猜想. | * P/NP问题 * 霍奇猜想 * 庞加莱猜想（已证明） * 黎曼猜想 * 杨-米尔斯存在性与质量间隙 * 纳维-斯托克斯存在性与光滑性 * 贝赫和斯维讷通-戴尔猜想 |. **黎曼猜想**（英语：Riemann hypothesis，RH）由德国数学家波恩哈德·黎曼于1859年提出。它是数学中一个重要而又著名的未解决的问题，有“猜想界皇冠”之称，多年来它吸引了许多出色的数学家",
    "具体来说，黎曼猜想的核心在于预测素数分布的偏差。素数定理虽然告诉我们素数的大致分布趋势，但仍有一定的误差。这种误差可以通过黎曼ζ 函数的零点来",
    "... 猜想（7）——非零区域与素数定理的余项. 黎曼猜想本身围绕着zeta函数展开。而在接下来的时间里，数学家们为了得到普适性的结论，开始研究zeta函数的一种推广——L函数。 等",
    "科目：数学知识点：条件概率、全概率公式与贝叶斯公式 公众号：摆渡考研工作室摆渡提供最优质的的课程与资料,提供经济学与数学同步辅导.",
    "Conditioning（条件）会改变样本空间，改变前后的样本空间里，A 发生的概率不变。 如果A 不是空集，则A 和B 一定不是互斥事件。 新的事件C 作为条件时，A 和B 的独立性会改变。",
    "條件概率是「在B的範圍內，A與B重疊部分占B的比例」，本質是樣本空間從Ω縮減為B後，A的概率重新歸一化（除以P(B)保證概率和為1）。 4.1.2 期望計數法：. 用計數",
    "正态分布有两个关键参数：数学期望（均值，μ）和方差（σ²），它们定义了分布的中心位置和宽度。 1. **正态分布的密度函数与分布函数** 正态分布的密度函数通常",
    "2.1 运动员选拔 · 2.2 随机变量的方差 · 2.3 (0-1)分布的方差 · 2.4 均匀分布的方差 · 2.5 方差的性质 · 2.6 二项分布的方差 · 2.7 正态分布的方差 · 2.8 协方差",
    "绝对收敛的时候，称数学期望存在。 对于数学期望，存在四个基本公式：. (1)对于常数C，E(C)=C；. (2)E(CX)=CE(X)；. (3)设有两个随机变量X,Y，那么E(X+Y)=E(X)+E(Y)；. (4)若两个",
    "切比雪夫不等式（Chebyshev's Inequality）是概率论中的一个重要不等式，用于描述随机变量的取值与其数学期望（均值）之间的关系。 定义. 设随机变量X的数学",
    "设随机变量具有数学期望. 方差. 则对于任意. 都有：. 定理的. 为：. 等价形式. 定理（切比雪夫不等式）： μ ε- μ μ ε+. ( ). f x. 6. 适用范围：对于期望、方差存在的随机变量. —",
    "棣莫弗-拉普拉斯中心极限定理：设随机变量～B(n,p)(n=1,2,…)，0<p<1，则的标准化随机变量依分布收敛于标准正态分布。 与大数定律相关的还有切比雪夫不等式：.",
    "[数理知识]参数估计：点估计、区间估计及置信区间. 参数估计是数理统计中 ... - 参数估计：学习点估计（如矩估计、极大似然估计）和区间估计，理解其优缺点。",
    "主要内容： 点估计： 矩估计极大似然估计点估计的评判准则区间估计： 置信区间符号说明： 1 参数估计问题2 点估计2.1 矩估计矩估计法的基本思想是根据大数",
    "点估计就是用一个数值对对总体参数给出估计；而区间估计是在点估计基础上，给定一个具体的估计范围。 例如，估计中国全部人口的平均身高。161cm就是一个点",
    "卡方检验是一种用途很广的计数资料的假设检验方法，主要是比较两个及两个以上样本率（构成比）以及两个分类变量的关联性分析。根本思想在于比较理论频数和",
    "专业上，p值为结果可信程度的一个递减指标，p值越大，我们越不能认为样本中变量的关联是总体中各变量关联的可靠指标。 p值是将观察结果认为有效即具有总体代表性的犯错概率。",
    "假设检验是统计学中一种常用的方法，用于判断关于总体参数的某个假设是否成立。下面是一些常见的假设检验方法的简要介绍：. t 检验（t-test）：.",
    "... 关系。 等价关系：如果R具有自反性，对称性，传递性，则称R是一个等价关系. 等价类. 定理1.2.7：. 划分：. 商集：设R是非空集合A上的等价关系，以R的所有不同等价",
    "等价关系（en:Equivalence relation）、集合划分（en:Partition of a set）、等价 ... 序理论（en:Order theory）：偏序关系或半序关系（en:Partially ordered",
    "本节介绍数学家门所谓的关系定义，并且研究在数学中常见的两种关系: 等价关系(equivalence relations)和全序关系(order relations)。 关系: 集合A 上的关系",
    "欧拉回路： 一条不重复边的路径，且起始和结束于同一个顶点。 它包含每个顶点和每条边。 哈密尔顿回路： 没有重复的边或顶点，除了第一个和最后一个顶点。",
    "1.8 有关性质若图G=〈V,E〉 具有哈密尔顿回路, 则对于结点集V 的每一个非空子集S 均有W(G-S)≤|S| 成立。 其中W(G-S) 是G-S中连通分支数。",
    "欧拉图：由于全偶度必定走出闭链，剥离并拼接闭链。 Ore 定理：任意两点度数和不小于点数推出哈密顿回路，是因为反证加边至再加一条就出现回路，利用",
    "本系列文章将介绍鸽巢原理、排列组合、二项式定理、容斥原理、生成函数与递推 ... 本章（鸽巢原理）将介绍组合数学中最简单、最基本的定理：鸽巢原理.",
    "容斥原理, Pi 为有某个禁止模式的排列. 生成函数. 定义. A(x)=a0+a ... 正整数拆分. 递推关系. 常系数线性齐次递推关系. 常系数线性非齐次. 迭代归纳法. 生成函数求解递推",
    "组合数学核心课件合集：涵盖排列组合、母函数、递推关系、容斥原理与鸽巢原理. n）、组合数C(n,k)的组合意义与代数性质（如对称性、Pascal恒等式",
    "... 可计算编码的概念扩展到可数之外。 这显然也涉及将可计算性扩展到自然数之外。 一种自然的方法是将广义可计算性视为在称为可容许集合的东西中的可定义性。",
    "编辑：一阶逻辑（包括一阶谓词逻辑）在语法上是不完整的。而简单的命题逻辑在这两种类型上都是完整的。",
    "哥德尔的第一条不完备定理表明任何一个允许定义自然数的体系必定是不完备的：它包含了不能在此体系内以一阶谓词逻辑形式证明的命题，并且该命题的否命题也不能在该体系内以一",
    "# 线性代数基础知识. |字数总计:4.7k|阅读时长:17分钟|阅读量:|评论数:. ## 标量、向量、矩阵、张量. 一个标量就是一个单独的数值，一般用小写的变量名表示。在定义标量时要说明标量所属的数据类型，如“另n∈N表示社团的数量”。. ### 向量. 可以简单理解为一列数，可以通过索引确定每个单独的数，通常使用粗体的小写表示。当我们需要明确表示向量中的元素是，一般都是表示成为列向量如下：.",
    "## 新浪教育. # 归纳总结：线性代数重点内容与题型. ## 万学海文. 有三角行列式、范德蒙行列式、行和或列和相等的行列式、三线型行列式、爪型行列式等等，必须熟练掌握相应的计算方法。. 矩阵是线性代数的核心，是后续各章的基础。矩阵的概念、运算及理论贯穿线性代数的始终。这部分考点较多。涉及伴随矩阵的定义、性质、行列式、逆矩阵、秩及包含伴随矩阵的矩阵方程是矩阵试题中的一类常见试题。有些性质得证明必",
    "# Ma Jia-Jun （马家骏）. # 高等代数与解析几何知识点. * 定义, 常见的例子: \mathbb{R}^n, 多项式, 连续函数等组成的空间…. * 从已知线性空间构造新的线性空间: 子空间, 商空间, 直和, 线性映射的核空间/像空间, (线性映射组成的空间\mathop{Hom}(V,W), 对偶空间V^*= \mathop{Hom}(V,K), 直积, 张量积, 外积… ).",
    "基与维数. 基是向量空间中一组线性无关的向量，通过线性组合可以表示出 ... 核是指使得线性变换结果为零向量的所有输入向量的集合，核是一个子",
    "       x1 x2 . 证 设V 是数域F 上的线性空间,",
    "... 知识#物理#数学. ... 【小旭学长】用最直观的方式告诉你：什么是奇异值分解SVD--SVD如何分解时空矩阵.",
    "奇异值分解，singular value decomposition，通常简写为SVD分解。 备注：建议看这部分知识的小伙伴可以先看矩阵的LU分解，QR分解。 已知正定矩阵（",
    "SVD 的定义 奇异值分解（Singular Value Decomposition, SVD） 是一种将矩阵分解为三个部分的方法，适用于任意的矩阵。给定一个m \times n 的矩阵A ，SVD",
    "在R\mathbb RR 中，两个向量内积（Inner product）和点积（Dot product）是同一个概念，也就是内积与点积一样。 ... 正交的向量张成的子空间上投影向量也更容易。",
    "施密特正交化主要利用了矢量点积的投影性质，可以将矢量分解为平行和垂直两个矢量。以下我们分别以二维向量空间和三维向量空间为例详细介绍施密特正交化的",
    "* {\displaystyle \langle {\boldsymbol {v}}_{1},{\boldsymbol {v}}_{2}\rangle }：{\displaystyle {\boldsymbol {v}}_{1}}**与{\displaystyle {\boldsymbol {v}}_{2}}的内积**. * {\displaystyle \mathrm {span} \{{\bo",
    "◦ 一般情况下,部分主元高斯消去(LU分解)的相对残差很小. ◦ 若问题不病态(矩阵条件数不很大)，且使用部分主元高斯. 消去法，则将得到很准确的解. ◦ 若不选主元，则相对残差、解",
    "在高斯消元法中，我们通过初等行变换将矩阵A 转换成上三角矩阵U，同时在消元步骤中用到的乘子可以整理成一个下三角矩阵L，使得最终有A = LU\。也就是说，LU",
    "LU分解的复杂度为三阶，但只需在预处理阶段进行分解一次；对于不断变化的b ，只需要求两次三角形线性方程组Ly=b 和Ux=y ，总体复杂度为二阶，求解效率显著提高。 3 列主元Gauss",
    "人体解剖学基础 上肢 下肢 脊柱与背部 胸部 腹部 骨盆与会阴 头颈部 神经解剖学 医学影像解剖学. 介绍 肌肉系统 神经系统 心血管系统 消化与吸收. 人体解剖学基础 上肢 下肢 脊柱与背部 胸部 腹部 头颈部 神经解剖学. | 局部解剖学 | 将人体分为若干个区域进行研究，包括：上肢、下肢、躯干（胸、腹、盆腔、背部）、头颈部以及神经解剖学。 |. | 系统解剖学 | 按器官系统研究人体结构，包",
    "[人体解剖学基础知识点整理（重点）] 人体解剖学是研究人体的结构，而生理学是研究人体的功能。人体结构非常复杂，所以解剖学内容包含不同的层次，从最小的细胞到最大的器官，以及器官之间的关系。",
    "[最全知识点总结！人体解剖学，高清教学视频（175课） - 搜狐] 在整个医学科中，人体解剖学是一门重要的基础课程，其任务是揭示人体各系统器官的形态和结构特征，各器官、结构间的毗邻和联属，为进一步学习后续的医学",
    "[生理学-第四章-循环系统 - 思维导图] # 生理学-第四章-循环系统. 编辑于2021-10-20 21:47:05. 护血管壁完整性的功能。 2.参与生理止血功能。 (1)血小板粘附、聚集形成松软止血栓,防止出血。 (2)血小板分泌ADP、5-羟色胺、儿茶酚胺等活性物质,ADP使血... 生理学思维导图，全面覆盖人体各大系统的工作原理，包括循环系统、呼吸系统、消化系统、神经系统等核心知识点。",
    "单元1：人体解剖学和生理学 · 人体循环系统简介 · 呼吸系统简介 · 泌尿系统简介 · 血液系统简介 · 免疫系统简介 · 胃肠道系统简介 · 神经系统简介 · 人体肌肉系统简介.",
    "[人体的基本结构 - 人卫助手] 2.2.2 呼吸系统呼吸道：鼻、咽、喉、气管、支气管（过滤、温暖空气）。 肺：气体交换的核心，肺泡通过扩散作用实现氧气与二氧化碳的交换。 2.2.3 神经系统中枢神经系统：脑（大脑",
    "[炎症与肿瘤相关性研究进展与展望 - Baishideng Publishing Group] 在过去的十年中, 有明确的证据表明, 炎症在肿瘤发生中起着关键作用. 肿瘤外源性炎症是由许多因素引起的, 包括细菌和病毒感染、自身免疫性疾病、肥胖、",
    "[细胞凋亡与癌症信号传导通路] 细胞凋亡，又称程序性细胞死亡，是一种通过清除受损或不需要的细胞来维持组织稳态的过程。 在癌症中，这一过程经常失效，导致异常细胞和转化的细胞可以存活繁殖。",
    "## Search. # 细胞凋亡. 正常的组织中存在一种细胞生成与损失的平衡，这种平衡通过细胞分裂和细胞死亡达到。老的细胞会逐渐受损并被淘汰，这是细胞重建的一种重要形式。例如皮肤细胞的脱落和消化道细胞的更替。像细胞分裂一样，细胞死亡也被严格地控制。细胞经常通过程序性细胞死亡或细胞凋亡而死去。1 细胞凋亡是细胞内的“自毁”按钮。. 凋亡是一个非常有序的过程。在此期间基因组被分解，细胞分裂成更小的碎",
    "[药理学- 01 - 基础知识] 药理学是研究药物与机体相互作用规律的一门科学。这门学科的主要目的是：. 阐明elucidate药物的作用规律，为临床合理用药提供理论基础；; 研制开发新药，",
    "药物作用的特异性，指的是药物在生物体内能够特异性地与某一特定的生物分子或受体结合，进而引发特定的药理效应。这种特异性主要取决于药物的化学结构。例如",
    "[[PDF] 《药理学》教学大纲 - 基础医学院] 《医学机能实验学》.北京：科学技术出版社，2013 年10 月 七、教学内容（理论） 第一章 药理学总论-绪言 【教学内容】 药理学的任务；药理学、药效学、药动学和药物的基本概念；药理学的发展史、我国本草的贡 献及国内外药理学的新进展；新药开发与研究。 【教学基本要求】 掌握药物、药理学、药物效应动力学及药物代谢动力学概念。了解药理学研究对象、任",
    "[免疫学基础知识(一)：免疫系统概述 - 知乎专栏] · 先天免疫是由宿主生殖系基因编码的受体激活的一组固有反应，这些受体识别许多外来物质共享的分子模式，这些外来物质不存在于哺乳动物宿主中。 · 适应性免疫是对独特的外来",
    "[认识免疫：常见的免疫细胞+免疫分子(基础篇) - MCE] 树突状细胞几乎存在于所有组织中，它们检测体内平衡失衡并处理抗原以呈递给T 细胞，从而建立先天免疫反应和适应性免疫反应之间的联系。 树突状细胞有两种不同的功能状态："",
    "[免疫学基础知识 - Chen Dianyu] 免疫学基本概念等背景知识介绍。 免疫器官. 免疫组织（immune tissue）又称为淋巴组织（lymphoid tissue），在人体中广泛分布，其中肠胃、呼吸道、泌尿生殖",
    "[心血管系统生理学- 分析心脏生理- 血压调节 - Flashcards World] 心血管系统生理学. 分析心脏生理、血液循环动力学、血压调节、微循环和组织灌注的机制，以及心血管系统的整合调控。 心血管系统的主要组成部分有哪些？ Click to flip.",
    "[视频: 心血管系统的调节 - JoVE] 体内平衡。 心血管系统的调节涉及自主神经系统（ANS）、压力感受器和化学感受器，确保心率和血压得到适当调节，以响应不同的生理 ... 13.3 : 心脏生理学：心脏",
    "[心脏生理介绍血压调控模型Starling定律 - University of Toronto] 心脏生理. 本网站关于心脏麻醉的心脏生理学部分涵盖了心脏生理的一些基本原理。 第一个模型是用来阐述关于血压的调控。进入此模型可以点击这里 或者点击下图，也可以",
    "# 呼吸系统的气体交换过程：深入解析肺通气与肺换气. 2025-07-06 13:25 | 温柔系煮妇. 在之前的分享中，我们探讨了生理学的多个方面，包括细胞的基本功能、血液以及循环系统。现在，我们将目光转向呼吸系统，探讨其核心概念——呼吸。呼吸，简而言之，就是机体与外界环境之间进行的气体交换过程。. #### ◆ 概念与动力. **肺通气**是指肺与外界环境之间进行的气体交换过程。这一过程的主要",
    "# 氧气和二氧化碳的交换. 作者：Rebecca Dezube, MD, MHS, Johns Hopkins University. Albert, MD, Department of Medicine, University of Colorado Denver - Anschutz Medical. 已审核/已修订 5月 2025 | 修改的 7月 2025. 呼吸系统的主要功能是吸入氧气和",
    "（2）气体在血液中的运输；. （3）内呼吸。 外呼吸是大气与肺进行气体交换以及肺泡与肺毛细血管血液进行气体交换的全过程；内呼吸指的是血液与组织细胞间的气体交换，而",
    "[神经传递 - 神经系统疾病 - MSD诊疗手册专业版] # 神经传递. 动作电位沿轴突的传播是电性的，由钠离子和钾离子穿过轴突膜交换引起的。神经元每次受刺激后会产生相同的动作电位，并按某一固定速度沿轴索传导。其速度取决于轴索的直径和其周围的髓鞘程度，小的无髓纤维的传导速度在1～4m/s，而较粗大的有髓纤维的传导速度可达75m/s。有髓纤维的神经传导速度较快，因为其存在按一定间隔分布的无髓鞘包裹的",
    "[神经传导和突触传递 - 知乎专栏] 当神经元发放动作电位时，就会从神经终扣释放出叫做神经递质的化学物质。神经递质在突触间隙扩散，并和下一个神经元细胞膜上特殊的受体分子发生相互作用。",
    "在大多数情况下，神经递质从突触中被迅速去除，这要归功于 ... 这些分子是非常规的，因为它们不被储存在突触囊泡中，并且可能将信息从突触后神经元传递到突触前神经元。",
    "[激素调节_百度百科] # 激素调节. 激素调节是指由内分泌器官（或细胞）分泌的化学物质通过体液运输，对机体新陈代谢、生长发育等生理活动进行的调节方式，属于体液调节范畴，具有微量高效性、通过血液循环运输、特异性作用于靶器官或靶细胞等特点。其调节过程包括血糖平衡调节、甲状腺激素分泌的分级调节等实例，涉及反馈调节机制 [1]。. 20世纪前学界普遍认为生理活动仅由神经系统调控。1902年斯他林与贝利斯",
    "[激素间的相互调节与影响- 内分泌学 - 天山医学院] 机体内的任何一种激素的合成和分泌都必须受到另一种（些）激素的调节，除调节“轴”内反馈环激素的调节作用外，其他激素也往往直接或间接影响其分泌。激素间的调节现象可归纳为两",
    "[内分泌系统概述 - 内分泌及代谢紊乱 - MSD诊疗手册专业版] # 内分泌系统概述. 内分泌系统通过激素来协调不同器官的功能；激素由内分泌腺体（无腺管）内特定类型细胞分泌释放到血循环中的化学物质。激素进入血循环后，作用于它们的靶器官，靶器官可以是其他的内分泌腺或终末器官。中的细胞（旁分泌），另一些激素甚至作用于同类型的细胞（自分泌）。. 激素选择性地结合于靶细胞内部或表面的受体。激素与细胞内的",
    "[[PDF] …… 一… 肿瘤转移抑制基因的作用及临床意义] · 142· 实用临床医药鸯志 2011年第15卷第7期 J ournal of CI i ni cel M edi ci ne i n Pract i ce ⋯⋯ 一⋯ ⋯ 肿瘤转移抑制基因的作用及临床意义 管孝臣1，秦文星2，颜荣林1 ( 上海长征医院，1．普外科；2．肿瘤科，上海，200090) 关键词：肿瘤转移抑制基因；转移；临",
    "[肿瘤生物学：（6）抑癌基因 - 知乎专栏] 通过这种方式，突变的抑癌基因，就有可能从一条染色体转移至它的同源染色体，从而代替原本在这一位点上的野生型等位基因，这种基因转换的发生频率在每",
    "[肿瘤转移抑制基因-1与肿瘤关系的研究进展] 肿瘤转移抑制基因-1(tumor metastasis suppressor gene-1, TMSG-1)是一个新发现的肿瘤转移抑制基因, 在促进肿瘤细胞凋亡和抑制肿瘤细胞浸润和转移中起着重要的作用.",
    "> 力不是维持物体的运动（速度）的原因。一旦物体具有某一速度且不受外力，就将保持这一速度匀速直线地运动下去。",
    "- **核心思想**：物体在不受外力时，将保持静止或匀速直线运动状态。",
    "![](https://bkimg.cdn.bcebos.com/formula/97156d96df427ff8b4d29e7afd5defe3.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/3161545fe3bb812c4bdc0241036f78a4.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/b2dbc678482ff7070f3f20cf90d88cf7.svg)",
    "- **内容**：物体的加速度与所受合外力成正比，与质量成反比，加速度方向与合外力方向相同。",
    "> 由两火柴头飞出的路程大致相等，可说明物体间的作用力是相互的。",
    "- **内容**：两个物体之间的作用力和反作用力总是大小相等，方向相反，作用在同一条直线上。",
    "![](https://bkimg.cdn.bcebos.com/formula/8e844f6d30d8166349d602762355c548.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/21a8548c5eacb7703b922f65f5da49c3.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/dcfee74edc9c2319ee08fbd9f3ef92e2.svg)",
    "*（附：倾斜滑动法和水平拉线法示意图、倾斜导轨法示意图、牛顿第二运动定律非线性拟合图）*",
    "![](https://bkimg.cdn.bcebos.com/formula/5d9151f9718fd47e1679dbce35b56f12.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/119e0aec626516e7abf1bd4271d87616.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/2a7f4d616d4c16dbf634e349d641b055.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/4f88ebed045531c75bb193bac407450c.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/59a1ed349bda06d7c092f267a686aa94.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/15e991fc80b6ce821eb090b1981d0abc.svg)",
    "![曲线运动图示](https://bkimg.cdn.bcebos.com/smart/e1fe9925bc315c6034a84c10fde8dc1349540823d981-bkimg-process…)",
    "![](https://bkimg.cdn.bcebos.com/formula/6b309b084f2136970f1b000bfe744b28.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/bbb056859182573bf7af7d76ce6ecdbd.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/9ba8a986e51d256bd18f610ea83c0a8e.svg)",
    "![简谐运动图示](https://bkimg.cdn.bcebos.com/smart/377adab44aed2e738bd434ccf758b68b87d6267f4c83-bkimg-process…)",
    "![](https://bkimg.cdn.bcebos.com/formula/7cc40825ec2548d8d13ab73d5c2bbe70.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/44c18d9b097ac3068ca67ed425bdb111.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/f164bd4166b63799aaefbbafe4f54f75.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/27dcd6559d22e5048a10ee910bed3616.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/cb40def24c2692dd95a96f218359630b.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/03bd5738646a439d4ae0e52453e8c53c.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/fc39f411d298932b69fc5c227de15b0f.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/fa5f1a2e5fdcde68bbc0d7330df362b6.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/21f0df8c854b31083906770037fd2a48.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/34bd44b53509b036bc30c29de14a9e6e.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/e6bb5037d1f428df3d5d836963ba83e6.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/4eddbf7998c8a835b66f156584c05389.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/e0981a69395c002fc65f89c941d84f89.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/a18c74052268fdcc1753950a3b3389a9.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/9ea67bfb9ff52345d0c2da83a62e0a34.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/8000efde8b42ca76ee31212f7853d175.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/f955e89f9173c0a30fe9c1278059af21.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/bbb985fee25eb9e8c5b40591bcc37327.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/a6b42150deb7253d72e189e7c3468ea8.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/6d41535e41d475e9a25e0480f58ba640.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/f204af598ebb51ef78e374b2966aa369.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/9cfde8a5cffda3d019c188df332728a7.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/ea956747819a5b67a5f53df1433be8d3.svg)",
    "![](https://bkimg.cdn.bcebos.com/formula/6c5b2898ba02d07c819efe7d1bd20f6e.svg)",
    "![](https://bkimg.cdn.bcebos.com/for",
    "[... summary truncated for context management ...]",
    "> **病理学**是研究人体疾病发生的原因、发生机制、发展规律以及疾病过程中机体的形态结构、功能代谢变化和病变转归的一门基础医学科学。",
    "1. **病因学（etiology）**：疾病发生的原因，包括内因、外因及相互关系",
    "2. **发病学（pathogenesis）**：病因作用下疾病发生、发展的具体环节和机制",
    "3. **病理变化（lesion）**：机体功能代谢和形态结构改变，及其与临床表现的关联（临床病理联系）",
    "- 诊断病理学（外科病理学）：以诊断为目的，包括尸检、活检、细胞学",
    "- 人体病理学（材料来自患者）与实验病理学（来自实验动物、组织培养等）",
    "以已学过的正常生理学科为基础，解释疾病状态下的形态、功能、代谢改变，并关联临床症状、诊断和预后。",
    "> 病理诊断是在观测器官大体改变、镜下组织结构和细胞病变特征而做出的，比临床分析性诊断及影像学诊断更具客观性和准确性，被视为带有“宣判性质的诊断”，国外称病理医生为“doctor’s doctor”。",
    "> 然而病理诊断也有固有局限性，需提高技术并加强临床病理沟通。",
    "现代病理学已深入亚细胞、蛋白表达及基因水平，方法渗透到基础医学、临床医学、预防医学和药学等领域，成为判断新技术、新药效毒理的重要依据。",
    "- 直接观察病理改变，明确诊断与死因，验证临床诊疗，发现传染病、地方病，积累人体病理材料。",
    "- 通过局部切除、钳取、穿刺等方法获取活体病变组织，进行组织学、组织化学、超微结构等研究。",
    "- 优点：组织新鲜，保留病变真像，利于及时准确诊断和疗效判断，对肿瘤等预后极有意义。",
    "- 在动物身上复制人类疾病模型，进行连续性取材，研究病因、发病机制及药物影响。",
    "- 体外培养组织或细胞，观察肿瘤生长、癌变、病毒复制、染色体变异等，并可施加外界干预。",
    "- **大体观察**：肉眼＋放大镜、量尺等，观察病变大小、形态、色泽、质地等。",
    "- **组织学观察**：切片染色后光镜观察，最常用，可诊断疾病。",
    "- **细胞学观察**：采集脱落细胞或穿刺细胞制成涂片，常用于肿瘤早期诊断（如肺癌、宫颈癌等），但取材局限有时影响准确度。",
    "- **超微结构观察**：透射/扫描电镜，可达亚细胞、大分子水平，将形态与机能代谢联系。",
    "- **组织/细胞化学观察**：显示组织细胞中蛋白质、酶、核酸、糖原等化学成分，可在形态改变之前检出变化。",
    "- **免疫组织/细胞化学**：了解免疫学性状，辅助研究和诊断。",
    "- **现代分子技术**：放射自显影、显微分光、图像分析、分析电镜、流式细胞术、PCR、分子原位杂交等，实现从定性到定量的飞跃，深化疾病研究深度。",
    "- 古希腊 Hippocrates（希波克拉底）提出“体液病理学”（四元素说→四种体液），主张疾病由体液失衡引起。",
    "- 罗马 Galen（盖伦）继承并发展体液学说，强调血液和“灵气”，其权威延绵近1500年。",
    "- 古代中国《黄帝内经》《诸病源候论》《洗冤集录》等对病理学有重要贡献。",
    "- 18世纪中叶，意大利 Morgagni 根据尸检创立 **器官病理学**，标志病理形态学开端。",
    "- 19世纪中叶，德国 Virchow 借助显微镜提出 **细胞病理学**，奠定现代病理学基础，将重心从体液转向固体（细胞）。",
    "- 徐育明、胡正详、梁伯强、谷镜汧、侯宝璋、林振纲、秦光煜等学者在教材编写、尸检活检推广、地方病/肿瘤/心血管病研究中卓有贡献，为中国病理学奠定基础。",
    "- 电子显微镜技术、细胞/分子生物学、免疫学、遗传学等推动病理学进入 **超微病理学、分子病理学、分子免疫学、分子遗传学** 等前沿领域。",
    "- 现代认识：几乎所有疾病均受遗传因素影响；免疫状态与许多疾病的发生发展密切相关。",
    "- 1543年 Vesalius《人体解剖学》出版，推翻盖伦许多错误，成为现代医学开端。",
    "- 1628年 William Harvey（哈维）发表《心血运动论》，建立血液循环理论，后由 Malpighi 发现毛细血管而完善。",
    "- 显微镜的发明使雷文霍克首次观察到血球、精子等，极大推进了细胞层面的认知。",
    "> **2021年**，浙江大学田梅、张宏提出“透明病理”概念：基于分子影像的分子识别和示踪优势，通过多尺度多模态分子影像与病理学融合，将机体生物特征全尺度“透明化”，实现无创在体评价，推动精准医学发展。",
    "- 病理学既是基础学科，又是临床诊断的实践学科，具有桥梁和判决性诊断地位。",
    "- 研究方法涵盖从大体到分子水平的完整链条，强调形态与功能相结合。",
    "- 学科发展史反映了人类从经验医学到实验科学，再到分子精准医学的认知演进。",
    "- 中国病理学具有独特疾病谱资源，需在提升尸检率、发展新技术和加强原创研究方面持续努力。",
    "[牛顿运动定律知识点归纳+典型例题总结，一份搞定物理难点！] 将牛顿运动定律与运动学知识结合可推导动量定理，动能定理，动量守恒定律和机械能守恒定律；将牛顿运动定律与万有引力结合，可研究天体运动规律；此外，牛顿运动定律在电磁学，",
    "1 牛顿运动三定律一、牛顿第一定律 (1)反映任何物体都具有惯性,牛顿第一定律又叫惯性定律。 (2)当物体受到其他物体作用（力）时才会改变其运动状态,即其他物体的作用是",
]


# ============================================================
# 中文维基百科摘要
# ============================================================
ZHWIKI_FILE = "/mnt/d/neuroflow-model/data/zhwiki_summaries.txt"
_zhwiki_cache = None
_zhwiki_index = 0

def get_zhwiki_summaries(limit=5):
    global _zhwiki_cache, _zhwiki_index
    if _zhwiki_cache is None:
        if os.path.exists(ZHWIKI_FILE):
            with open(ZHWIKI_FILE, "r", encoding="utf-8") as f:
                raw = f.read()
            _zhwiki_cache = [s.strip() for s in raw.split("\n---\n") if s.strip() and len(s.strip()) > 20]
        else:
            _zhwiki_cache = []
    if not _zhwiki_cache:
        return []
    items = []
    for i in range(limit):
        idx = (_zhwiki_index + i) % len(_zhwiki_cache)
        items.append(_zhwiki_cache[idx])
    _zhwiki_index = (_zhwiki_index + limit) % len(_zhwiki_cache)
    return items


# ============================================================
# kaikki 中文词典 — 顺序遍历缓存
# ============================================================
KAIKKI_ZH_CACHE = "/mnt/d/neuroflow-model/data/kaikki_zh_cache.jsonl"
_kaikki_zh_lines = None

def fetch_kaikki_zh(limit=3):
    """从缓存顺序读取，位置自动推进，耗尽后回绕"""
    global _kaikki_zh_lines
    pos_file = KAIKKI_ZH_CACHE + ".pos"
    if not os.path.exists(KAIKKI_ZH_CACHE):
        return []
    
    if _kaikki_zh_lines is None:
        with open(KAIKKI_ZH_CACHE, "r", encoding="utf-8") as f:
            _kaikki_zh_lines = [line.strip() for line in f if line.strip()]
    
    pos = 0
    if os.path.exists(pos_file):
        pos = json.load(open(pos_file)).get("pos", 0)
    
    items = []
    for i in range(limit * 3):
        idx = (pos + i) % len(_kaikki_zh_lines)
        line = _kaikki_zh_lines[idx]
        try:
            entry = json.loads(line)
            word = entry.get("head_templates", [{}])[0].get("expansion", "")
            pos_tag = entry.get("pos", "")
            glosses = []
            for sense in entry.get("senses", []):
                glosses.extend(sense.get("glosses", []))
            gloss = glosses[0] if glosses else ""
            if word and gloss and len(word) >= 2:
                items.append(f"{word} ({pos_tag}): {gloss}")
            elif word and len(word) >= 2:
                items.append(f"{word} ({pos_tag})")
        except:
            continue
    
    new_pos = (pos + limit * 3) % len(_kaikki_zh_lines)
    with open(pos_file, "w") as pf:
        json.dump({"pos": new_pos, "total": len(_kaikki_zh_lines)}, pf)
    
    return items[:limit]


# ============================================================
# Wikipedia 全标题 — 顺序遍历缓存
# ============================================================
WIKITITLES_CACHE = "/mnt/d/neuroflow-model/data/wikititles_cache.txt"
_wikititles_lines = None

def fetch_wikititles(limit=4):
    """从缓存顺序读取 Wikipedia 标题"""
    global _wikititles_lines
    pos_file = WIKITITLES_CACHE + ".pos"
    if not os.path.exists(WIKITITLES_CACHE):
        return []
    
    if _wikititles_lines is None:
        with open(WIKITITLES_CACHE, "r", encoding="utf-8") as f:
            _wikititles_lines = [line.strip() for line in f if line.strip() and line.strip() != "page_title"]
    
    pos = 0
    if os.path.exists(pos_file):
        pos = json.load(open(pos_file)).get("pos", 0)
    
    items = []
    for i in range(limit * 3):
        idx = (pos + i) % len(_wikititles_lines)
        title = _wikititles_lines[idx]
        if title and len(title) >= 2:
            items.append(f"Wikipedia article: {title}")
    
    new_pos = (pos + limit * 3) % len(_wikititles_lines)
    with open(pos_file, "w") as pf:
        json.dump({"pos": new_pos, "total": len(_wikititles_lines)}, pf)
    
    return items[:limit]


# ============================================================
# kaikki 英文词典 — 顺序遍历缓存
# ============================================================
KAIKKI_EN_CACHE = "/mnt/d/neuroflow-model/data/kaikki_en_cache.jsonl"
_kaikki_en_lines = None

def fetch_kaikki_en(limit=5):
    """从缓存顺序读取英文词条"""
    global _kaikki_en_lines
    pos_file = KAIKKI_EN_CACHE + ".pos"
    if not os.path.exists(KAIKKI_EN_CACHE):
        return []
    
    if _kaikki_en_lines is None:
        with open(KAIKKI_EN_CACHE, "r", encoding="utf-8") as f:
            _kaikki_en_lines = [line.strip() for line in f if line.strip()]
    
    pos = 0
    if os.path.exists(pos_file):
        pos = json.load(open(pos_file)).get("pos", 0)
    
    items = []
    for i in range(limit * 3):
        idx = (pos + i) % len(_kaikki_en_lines)
        line = _kaikki_en_lines[idx]
        try:
            entry = json.loads(line)
            word = entry.get("word", "")
            pos_tag = entry.get("pos", "")
            glosses = []
            for sense in entry.get("senses", []):
                glosses.extend(sense.get("glosses", []))
            gloss = glosses[0] if glosses else ""
            if word and gloss and len(word) >= 2:
                items.append(f"{word} ({pos_tag}): {gloss}")
            elif word and len(word) >= 2:
                items.append(f"{word} ({pos_tag})")
        except:
            continue
    
    new_pos = (pos + limit * 3) % len(_kaikki_en_lines)
    with open(pos_file, "w") as pf:
        json.dump({"pos": new_pos, "total": len(_kaikki_en_lines)}, pf)
    
    return items[:limit]


def get_knowledge_batch():
    batch = []
    counts = {"hn": 0, "github": 0, "npr": 0, "zhwiki": 0, "kaikki_zh": 0, "kaikki_en": 0, "wikititles": 0, "local": 0}

    hn = fetch_hn_top_stories(12)
    taken_hn = hn[:4]
    batch.extend(taken_hn)
    counts["hn"] = len(taken_hn)

    gh = fetch_github_trending(5)
    taken_gh = gh[:3]
    batch.extend(taken_gh)
    counts["github"] = len(taken_gh)

    npr = fetch_npr_headlines(6)
    taken_npr = npr[:3]
    batch.extend(taken_npr)
    counts["npr"] = len(taken_npr)

    zh = get_zhwiki_summaries(5)
    batch.extend(zh)
    counts["zhwiki"] = len(zh)

    kz = fetch_kaikki_zh(15)
    batch.extend(kz)
    counts["kaikki_zh"] = len(kz)

    ke = fetch_kaikki_en(25)
    batch.extend(ke)
    counts["kaikki_en"] = len(ke)

    wt = fetch_wikititles(20)
    batch.extend(wt)
    counts["wikititles"] = len(wt)

    needed = BATCH_SIZE - len(batch)
    if needed > 0:
        local_items = random.sample(LOCAL_KB, min(needed, len(LOCAL_KB)))
        batch.extend(local_items)
        counts["local"] = len(local_items)

    return batch[:BATCH_SIZE], counts


# ============================================================
# 向量编码 — 适配 TEXT_DIM 维度
# ============================================================
def encode(text, dim=TEXT_DIM):
    words = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)
    for i, w in enumerate(words[:200]):
        h = abs(hash(w)) % (2**31)
        for j in range(8):
            vec[int((h + j * 2654435761) % dim)] += 0.03 / max(len(words)/30, 1)
    vec += np.sin(np.linspace(0, np.pi * len(words)/15, dim)).astype(np.float32) * 0.08
    return vec / (np.linalg.norm(vec) + 1e-8)


def encode_batch(texts, dim=TEXT_DIM):
    """向量化批量编码 — 40 条 30× 快于逐个循环"""
    B = len(texts)
    vecs = np.zeros((B, dim), dtype=np.float32)
    for b, text in enumerate(texts):
        words = text.lower().split()[:200]
        nw = len(words)
        if nw == 0:
            continue
        scale = 0.03 / max(nw / 30.0, 1.0)
        for w in words:
            h = abs(hash(w)) % (2**31)
            for j in range(8):
                vecs[b, int((h + j * 2654435761) % dim)] += scale
        vecs[b] += np.sin(np.linspace(0, np.pi * nw / 15.0, dim)).astype(np.float32) * 0.08
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms


# ============================================================
# 主循环 — v4: 批量编码 + 多轮 SGD
# ============================================================
state = {
    "topics": 0, "train_steps": 0, "total_loss": 0.0, "avg_loss": 0.0,
    "knowledge": 0, "errors": 0, "started": datetime.now().isoformat(),
    "source_stats": {"hn": 0, "github": 0, "npr": 0, "zhwiki": 0, "kaikki_zh": 0, "kaikki_en": 0, "wikititles": 0, "local": 0}
}

if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        saved = json.load(f)
    state.update({k: saved.get(k, state[k]) for k in state if k != "source_stats"})
    state["source_stats"].update(saved.get("source_stats", {}))

# ════════════════════════════════════════════
# v3: 初始化 TrainableHead
# ════════════════════════════════════════════
model = create_multimodal(text_dim=TEXT_DIM, image_size=224, output_dim=OUTPUT_DIM, quantize=False,
                          hidden_dim=HIDDEN_DIM, memory_dim=MEMORY_DIM, num_layers=NUM_LAYERS)
head = TrainableHead(model, hidden_dim=HEAD_HIDDEN, n_actions=HEAD_ACTIONS, lr=0.01)

# 加载已训练权重 (优先 v4 新维度)
weights_src = WEIGHTS_FILE_V4 if os.path.exists(WEIGHTS_FILE_V4) else WEIGHTS_FILE
if os.path.exists(weights_src):
    try:
        weights = dict(np.load(weights_src, allow_pickle=True))
        head.load_weights(weights)
        print(f"[{datetime.now():%H:%M:%S}] 🔄 Loaded TrainableHead weights from {weights_src}")
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] ⚠ Load weights failed: {e}, starting fresh")

# 恢复训练统计
head.n_updates = state.get("train_steps", 0)
head.total_loss = state.get("total_loss", 0.0)

# 推理器 (使用 TrainableHead 作为模型 → 推理过程穿过可训练层)
reasoner = NeuroSymbolicReasoner(head)

# ════════════════════════════════════════════
# 获取知识
# ════════════════════════════════════════════
batch, source_counts = get_knowledge_batch()
network_count = sum(1 for t in batch if t not in LOCAL_KB)
print(f"[{datetime.now():%H:%M:%S}] Sources: {len(batch)} topics ({network_count} from web, {len(batch)-network_count} local)")

for src in ["hn", "github", "npr", "zhwiki", "kaikki_zh", "kaikki_en", "wikititles", "local"]:
    state.setdefault("source_stats", {})[src] = state["source_stats"].get(src, 0) + source_counts.get(src, 0)

# ════════════════════════════════════════════
# v5 核心：批量编码 + 批量梯度下降
# ════════════════════════════════════════════
x_batch = encode_batch(batch).astype(np.float32)  # [B, TEXT_DIM]
B = x_batch.shape[0]
batch_losses = []

for epoch in range(SGD_EPOCHS):
    # epoch 0: 原始向量; epoch 1+: 加噪声增强鲁棒性
    if epoch == 0:
        x_epoch = x_batch
    else:
        noise = np.random.randn(*x_batch.shape).astype(np.float32) * NOISE_STD
        x_epoch = (x_batch + noise) / (1.0 + NOISE_STD)

    # 收集全 batch 的 target 和 reward
    targets = []
    rewards = []
    for i in range(B):
        x = x_epoch[i:i+1]
        pred = head.predict(x)
        targets.append(int(np.argmax(pred.decision[0])))
        try:
            trace = reasoner.reason(x, max_steps=3)
            rewards.append(trace.final_confidence)
        except Exception:
            rewards.append(0.5)

    # 批量梯度下降 — 一次更新所有参数
    try:
        result = head.train_batch(x_epoch, targets, rewards)
        batch_losses.append(result["loss"])
    except Exception as e:
        state["errors"] += 1
        continue

    state["topics"] += B
    state["train_steps"] = head.n_updates
    state["total_loss"] = float(head.total_loss)

# ════════════════════════════════════════════
# 保存知识
# ════════════════════════════════════════════
fname = f"{state['topics']:06d}_knowledge.txt"
web_items = [t for t in batch if t not in LOCAL_KB]
text_snippet = (web_items[0] if web_items else batch[-1])[:2000] if batch else "no data"
with open(os.path.join(KB_DIR, fname), "w") as f:
    f.write(text_snippet)
state["knowledge"] = len([f for f in os.listdir(KB_DIR) if f.endswith('.txt')])

# ════════════════════════════════════════════
# 保存 TrainableHead 权重
# ════════════════════════════════════════════
try:
    np.savez_compressed(WEIGHTS_FILE_V4, **head.get_weights())
except Exception as e:
    print(f"  ⚠ Save weights failed: {e}")

# ════════════════════════════════════════════
# 记录训练统计
# ════════════════════════════════════════════
avg_loss = np.mean(batch_losses) if batch_losses else 0.0
state["avg_loss"] = round(float(avg_loss), 6)
state["last_run"] = datetime.now().isoformat()

# 计算适应度 (基于损失下降趋势)
if head.n_updates > 50:
    # 过去50步的平均 loss → fitness = exp(-avg_loss) in [0,1]
    recent_loss = float(head.total_loss / max(head.n_updates, 1))
    state["fitness"] = round(np.exp(-recent_loss), 4)
elif head.n_updates > 0:
    # 预热期: 显示当前 loss
    state["fitness"] = round(np.exp(-state["avg_loss"]), 4)
else:
    state["fitness"] = 0.5  # 初始

with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=2)

uptime = (datetime.now() - datetime.fromisoformat(state["started"])).total_seconds() / 3600
stats = head.stats()
print(f"[{datetime.now():%H:%M:%S}] "
      f"📖 {state['topics']} topics | 🎯 {state['train_steps']} steps | "
      f"📉 loss={avg_loss:.4f} | W_d={stats['W_d_norm']:.2f} W_v={stats['W_v_norm']:.2f} | "
      f"📈 fitness={state['fitness']:.4f} | ⏱ {uptime:.1f}h")
print(f"STATUS: topics={state['topics']} train_steps={state['train_steps']} "
      f"loss={avg_loss:.4f} fitness={state['fitness']:.4f} knowledge={state['knowledge']} errors={state['errors']}")
