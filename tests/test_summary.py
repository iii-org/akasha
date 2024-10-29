import pytest
import akasha.summary as summary
import os
from pathlib import Path


@pytest.fixture
def base_line():
    sum = summary.Summary(
        verbose=False,
        chunk_overlap=41,
        chunk_size=501,
        max_input_tokens=3020,
        temperature=0.15,
    )
    file_path = "./docs/mic/20230531_智慧製造需求下之邊緣運算與新興通訊發展分析.pdf"
    content = """
        邊緣運算定義與範疇
    鑒於市場客層多樣化、產品需求變動快速，少量多樣、彈性生產等需求，成為製造業
    者面臨的新常態，而加速製造場域的「智慧化」也成為眾業者當務之急。由於製造場
    域具備高度複雜性及產業差異性，不同的生產需求、製造流程、機台部署、人力配置、
    場域樓層規劃等，對於技術導入的需求皆不盡相同。
    近年邊緣運算、5G、Wi-Fi 6 以及 LPWAN 等新興通訊技術的發展，帶來低延遲
    （Latency）、高可靠度（Reliability）、大量設備連結的多樣應用契機。然而邊緣運
    算與新興通訊技術在智慧製造場域高度相關，邊緣運算的低延遲、高即時特性，需在
    高速的網路環境支持下才能發揮實際效益，兩者相互結合下，將使智慧製造的異質設
    備協作、AR／MR 巡檢、即時數據分析、預測性維護、遠距操作等不同類型的關鍵任
    務情境成為可能。
    從「數據流」角度定義「邊緣運算」
    根據工業物聯網聯盟（Industrial Internet Consortium, IIC）的定義來看，邊緣運算
    定義為「將運算靠近數據源，即數據產生或消耗的地點，目的為最小化延遲，節省網
    路頻寬，或滿足在處理地點附近執行的應用程式需求，例如安全性與合規性」，強調
    從數據流發生與消耗的位置，來定義邊緣運算。
    製造場域智慧化的核心是設備、人員、環境以及生產相關的「數據」（Data）。若從
    數據流的角度來看，大量、即時的類比（如波形）及物理訊號（如溫度、濕度、震動、
    壓力），從機器設備、環境端產生，為感測器（Sensor）所捕捉接收，進行類比數位
    轉換後，經由 IoT、邊緣節點（如邊緣伺服器）與網路設備，進入到地端或雲端伺服
    器進行儲存或批次處理的過程，會經過如下步驟：
    一、感測器（Sensor）
    感測器用於收集生產製造過程中的各種物理訊號，例如溫度、壓力、光敏、濕度、震
    動、位置等，進行類比到數位的基礎轉換，上述訊號可能由設備內建感測器模組（如
    AMR 移動狀態、機械手臂負載數據），或由既有設備外裝的感測裝置，傳出特定參
    數與數據。
    wsliou@iii.org.tw downloaded this document at 2023/8/10 13:57:44. © Copyright Market Intelligence & Consulting Institute.
    邊緣運算與新興通訊下的智慧製造應用展望 May 2023
    ©2023 Market Intelligence & Consulting Institute
    CDOC20230526001 2
    二、物聯網裝置（IoT Device）
    感測器轉換後的數據，藉由無線方式傳送到物聯網裝置，如智慧控制器或嵌入式
    （Embedded）硬體，並對數據進行初步處理和分析，並通常以無線方式與邊緣節點
    進行通訊。
    三、邊緣節點（Edge Node）
    物聯網裝置進行初步數據處理後，將數據傳送到較上層的邊緣節點，如邊緣伺服器
    （Edge Server）等。邊緣節點將進一步對數據進行分析，以減少向雲端傳輸的數據
    量。而部分 AI 模型亦可由雲端部署於邊緣節點，進行即時應用推論，降低往返的延
    遲時間。
    四、網路裝置（Networking Device）
    數據從邊緣節點經過網路裝置，如交換器（Switch）、路由器（Router）以及防火牆
    等，通過有線或無線網路（如 Wi-Fi 6 或 5G）連接到企業內或外部網路空間。在網
    路設備端，通常也會部署基礎資安控管機制。
    五、地端／雲端伺服器（On Premise / Cloud Server）
    經過網路傳輸後，數據最終到達工廠機房的本地雲（On Premise Server）或外部雲
    端伺服器（Cloud Server）。在地端或雲端伺服器上，數據可以進行長期儲存、備援、
    或進一步進行大量的批次分析和應用，例如大數據分析、機器學習模型訓練等較繁重，
    無法在邊緣端執行的運算工作。
    邊緣運算涵蓋範疇
    不同業者對於邊緣運算有不同定義，但在製造場域中，「邊緣運算」意指在物聯網裝
    置或邊緣節點的位置，就先行針對設備或環境傳入的數據，進行初步運算、分析或推
    理，並即時做出部分關鍵決策，迅速回饋至機器端執行，另一方面，即時性較無限制
    的數據，則會傳至更上層的地端、雲端伺服器，進行數據儲存、批次處理、模型訓練
    等工作。
    在製造場域端，需要進行邊緣運算的任務，通常包括即時監控和反饋、品質控制、生
    產速度調整和預測性維護等，藉由分析裝置即時數據，對潛在故障進行預測，提前警
    wsliou@iii.org.tw downloaded this document at 2023/8/10 13:57:44. © Copyright Market Intelligence & Consulting Institute.
    邊緣運算與新興通訊下的智慧製造應用展望 May 2023
    ©2023 Market Intelligence & Consulting Institute
    CDOC20230526001 3
    示以減少意外停機或機件損毀，降低維修成本並提高生產效率。此外，邊緣結點亦可
    藉由部署機器學習和 AI 模型，進行即時決策優化。如將深度學習模型，部署於自動
    視覺檢測（AOI）設備，以即時偵測複雜元件的缺陷。
    此外，如視覺巡檢、人員機器間的協同控制（如同一條生產線的 AMR、AOI 與機械
    手臂）、工廠自動化、遠距操作等應用場景，也有賴於低延遲、高可靠的邊緣運算，
    以避免碰撞、流程中斷，操作錯誤等情形發生，達到更高效的生產目標，以下為常見
    智慧製造應用案例的低延遲應用需求：
    表一、不同智慧製造應用的最低延遲容許度
    智慧製造應用案例 端對端容許延遲（毫秒/ms） 資料頻寬需求 超出延遲之影響
    MR / AR
    視覺擴增巡檢
    5-10 ms 高
    低-中
    （資料顯示不全）
    AMR 應用
    人機安全協作
    few ms 低-中
    高
    （機件碰撞、人員受
    傷）
    工廠自動化
    異直設備協作
    1-10 ms 低-中
    中
    （自動流程中斷）
    遠距操作
    遠距控制
    10-40 ms 高
    中-高
    （操作錯誤、機件損
    毀）
    資料來源： Latencytech、Ericsson，MIC 整理，2023 年 5 月
    最後，邊緣運算也可用於節省數據傳輸流量。經由在邊緣節點上對數據進行過濾
    （Filter）和聚合（Aggregation），可減少需要傳輸到雲端的數據量，降低數據傳輸
    成本。此外，邊緣運算亦更適合在本地端對敏感數據進行加密和處理，保護數據安全
    和隱私，對於需要嚴守數據保護法規的製造業企業，具有重要意義。
    wsliou@iii.org.tw downloaded this document at 2023/8/10 13:57:44. © Copyright Market Intelligence & Consulting Institute.
    邊緣運算與新興通訊下的智慧製造應用展望 May 2023
    ©2023 Market Intelligence & Consulting Institute
    CDOC20230526001 4
    邊緣運算於智慧製造發展現況
    全球結合邊緣運算與 5G 應用的智慧製造市場快速發展，需求領域則聚焦於 AMR 與
    自動控制、AR/VR 裝置應用、物聯網設備控制以及 IT/OT 整合（如預測性維護、故
    障預警）等應用。以下將從近年資訊技術商（IT）、營運技術商（OT）、雲端服務商
    （CSP）的邊緣運算技術及解決方案，以及邊緣運算不同環節的技術發展焦點，剖析
    邊緣運算於智慧製造發展現況。
    資訊技術商（IT）
    NVIDIA
    晶片大廠 NVIDIA 為近年邊緣伺服器的重要業者之一，尤以嵌入式邊緣系統 Jetson
    系列以及邊緣 AI 伺服器 EGX 系列為主要代表，產品包括小型 IoT 嵌入式 AI 應用的
    Jetson Nano，到企業級的邊緣 AI 伺服器平台，如內建 A100 GPU 的 EGX A100，
    最新產品則為 2022 年 9 月推出的 IGX 邊緣平台，強調高即時性的數位雙生（Digital
    Twin）和工業元宇宙支援。NVIDIA 發展關鍵優勢在於高效 AI 和深度學習（Deep
    Learning）功能，且具較完整泛用的軟體開發生態系（如 CUDA、TensorRT）。
    IBM
    IBM 在邊緣運算的產品布局，主要包括可將 IoT 工作負載，分配至邊緣裝置上進行部
    署及更新，以節省流量和管理成本的 IBM Edge Application Manager，以及能讓企
    業在任何邊緣位置執行 AI 應用的 IBM Watson Anywhere。IBM 邊緣運算產品和服
    務的特點，在於可自動化在大量邊緣裝置上執行、管理工作負載，並且提供深度 AI
    分析和 IoT 整合，以執行即時的邊緣決策和動作。此外，IBM 於 2022 年 12 月發布，
    與 Boston Dynamics 合作開發的邊緣 AI 視覺系統，可讓四足型態的機械載具在複
    雜或危險工業環境中，透過邊緣伺服器無線連接，進行精確的視覺分析辨認物件，並
    進行即時動作反饋。
    營運技術商（OT）
    Schneider
    工控大廠 Schneider Electric 在邊緣運算的產品，主要以 EcoStruxure 為代表，其為
    IoT 專用的、開放、具高度互操作性（Interoperability）的系統架構平台，主要功能
    wsliou@iii.org.tw downloaded this document at 2023/8/10 13:57:44. © Copyright Market Intelligence & Consulting Institute.
    邊緣運算與新興通訊下的智慧製造應用展望 May 2023
    ©2023 Market Intelligence & Consulting Institute
    CDOC20230526001 5
    為連接異質設備進行邊緣控制，並整合上層應用、分析及服務，EcoStruxure Edge
    Control 則將即時邊緣監控、數據處理 AI、機器學習結合，可用於辨識潛在的設備故
    障、安全風險等問題。此外，Schneider 亦提供超融合（Hyperconvergence
    Infrastructure）的數據中心 EcoStruxure Micro Data Center 解決方案，可在嚴酷
    的環境中執行即時數據處理和分析的邊緣應用。
    Honeywell
    Honeywell 的應用平台 Honeywell Forge 具有從邊緣到雲的運算能力，能在邊緣端
    進行的數據處理和分析，並將重要訊息推送到雲端，進行進一步的分析和優化，期解
    決方案主要用於即時設備健康監控、維護管理、能源管理等，並可以透過機器學習進
    行預測性維護。此外 Honeywell 的 Experion PKS （Process Knowledge System）
    工業自動化系統，可收集和處理來自工廠各處的邊緣數據，並在邊緣端進行即時控制
    和優化。並提供整合的應用介面，使營運工程師可從單一地點監控和管理整個工廠的
    數據運作狀態。
    雲端服務商（CSP）
    Google
    Google 於智慧製造的邊緣運算產品為 Google Cloud IoT Edge，以及 2022 年推出
    的 Google Cloud for Manufacture。Google 的優勢在於其強大的數據分析能力，
    並提供豐富的資料分析工具，如 BigQuery、Dataflow 等，可讓企業在如預測性分析、
    製造營運流程優化、即時報表等應用上較能輕易存取。而其開放原始碼的軟體生態系
    統，也允許企業在一定程度上進行需求客製化。此外，Google 亦逐漸將 Edge TPU
    （專為 TensorFlow Lite 模型設計的硬體加速器）導入產品，讓邊緣裝置可執行已訓
    練好的模型，強化邊緣 AI 的執行效率。
    Microsoft
    作為雲端與軟體業者，微軟主要的邊緣伺服器產品包括Azure Stack Edge以及Azure
    IoT Edge。同樣以混合雲協同運作效率為主要關注點，微軟的邊緣方案訴求工作負載
    與管理的平衡，可讓邊緣設備進行機器學習、推論及其他分析和處理任務，並將數據
    傳送到 Azure 雲端進一步處理、或進行訓練模型更新。而 Azure IoT Edge 則是專為
    物聯網設計的服務，其由容器構成，可靈活擴充功能，並支援部署 Azure Machine
    wsliou@iii.org.tw downloaded this document at 2023/8/10 13:57:44. © Copyright Market Intelligence & Consulting Institute.
    邊緣運算與新興通訊下的智慧製造應用展望 May 2023
    ©2023 Market Intelligence & Consulting Institute
    CDOC20230526001 6
    Learning 和 Azure Cognitive Services 的人工智慧模型，讓邊緣位置設備獲得推論
    能力。
    AWS
    AWS 在邊緣運算產品線上，包括能讓使用者在地端執行 AWS 架構，將 AWS 基礎服
    務、操作模式「複製」到邊緣伺服器的 AWS Outposts，以及在邊緣環境下最佳化 5G
    延遲，以支援 AR／MR 巡檢、IoT 即時應用的 AWS Wavelength。AWS 發展邊緣運
    算的主要優勢與特色，在於邊緣與 AWS 雲端服務的緊密接合（如 S3、EC2），讓「邊
    緣+遠端」的混合雲（Hybrid Cloud）數據管理更一致、應用情境更統一。
    邊緣運算於智慧製造技術發展焦點
    感測層：感測器與物聯網裝置，降低功耗為重要考量
    由於感測器需要大量布建於設備、工廠環境中，電池更換週期長，並通常須維持不斷
    電感測，因此低功耗一直是 Sensor 與 IoT 設備的重要議題。除了搭載小型太陽能面
    板外，近年利用收集機械運轉的震動能量，以維持裝置蓄電力的技術亦開始出現，而
    低功耗元件（如 ARM Cortex-M 系列的低功耗 MCU、MRAM、FeRAM 等低功耗記
    憶體）開始在 IoT 設備上普及採用，以及 2022 年開始利用微型 AI 晶片進行微機器
    學習（TinyML），預測使用峰值與離線時間的技術也逐漸成熟，可使 IoT 設備進行
    預測，並可在離峰時段進入休眠模式，進行節能。此外，近年亦出現高度整合多功能
    感測器，能在單一裝置同時監測多種物理特性，如溫度、壓力、濕度等，從而降低成
    本並簡化部署複雜度，同時提供較高的部署彈性。
    運算層：邊緣 AI 晶片普及，提升生產靈活性
    邊緣伺服器為組成邊緣運算的重要核心。近年因 AI 即時應用需求增加，如少量多樣
    需求下，需針對多樣、未經模型訓練的工件、產品快速進行視覺辨識，或進行異質設
    備（如不同品牌的 AMR、機械手臂）的生產流程協同應用、設備的預測維護等，在
    邊緣伺服器搭載 AI 晶片，進行邊緣推論已逐漸成趨勢，如搭載 NVIDIA GPU 的
    NVIDIA Jetson、搭載 TPU（Tensor Processing Unit）的 Google Coral 邊緣平台、
    聚焦視覺 AI 模組的 Intel Movidius Myriad X。除此之外，容器技術（如 Docker 和
    Kubernetes）和微服務架構在邊緣伺服器上的應用也逐漸普及，使應用程式能進行
    快速部署、擴展和維護，提高邊緣伺服器的資源利用率和靈活性。
    """
    return sum, file_path, content


@pytest.mark.summary
def test_Summary(base_line):
    sum, file_path, content = base_line

    assert sum.verbose == False
    assert sum.chunk_size == 501
    assert sum.chunk_overlap == 41
    assert sum.max_input_tokens == 3020
    assert sum.temperature == 0.15

    text = sum.summarize_articles(articles=content,
                                  summary_type="map_reduce",
                                  summary_len=500)

    assert type(text) == str

    sum_path = Path("summarization/")
    if not sum_path.exists():
        sum_path.mkdir()

    text = sum.summarize_file(
        file_path=file_path,
        summary_type="refine",
        summary_len=100,
        output_file_path="./summarization/summary.txt",
    )
    assert type(text) == str

    return
