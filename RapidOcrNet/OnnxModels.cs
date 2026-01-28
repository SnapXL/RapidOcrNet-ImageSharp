
namespace RapidOcrNet;

public enum ModelVersion
{
    V4,
    V5
}

public record ModelInfo(string Name, string Url);

public record RecognitionModelInfo(string Name, string Url, string? KeyUrl = null) : ModelInfo(Name, Url);

public record OcrSuite(
    string Name,
    ModelVersion Version,
    ModelInfo Detector,
    ModelInfo Classifier,
    RecognitionModelInfo Recognizer
);
public record ModelSource(string Name, string Url, string? KeyUrl = null);

public record OcrModel(
    string DisplayName,
    ModelVersion Version,
    ModelSource Detector,
    ModelSource Classifier,
    ModelSource Recognizer
);

public static class OnnxModels
{
    private static readonly ModelSource DefaultCls = new(
        "ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ch_ppocr_mobile_v2.0_cls_infer.onnx"
    );

    private static readonly ModelSource DetV5Mobile = new(
        "ch_PP-OCRv5_mobile_det.onnx",
        "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ch_PP-OCRv5_mobile_det.onnx"
    );

    private static readonly ModelSource DetV5Server = new(
        "ch_PP-OCRv5_server_det.onnx",
        "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ch_PP-OCRv5_server_det.onnx"
    );

    public static class V5
    {
        public static OcrModel ChineseMobile => new("Chinese V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("ch_PP-OCRv5_rec_mobile_infer.onnx", "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ch_PP-OCRv5_rec_mobile_infer.onnx"));

        public static OcrModel ChineseServer => new("Chinese V5 Server", ModelVersion.V5, DetV5Server, DefaultCls,
            new("ch_PP-OCRv5_rec_server_infer.onnx", "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ch_PP-OCRv5_rec_server_infer.onnx"));

        public static OcrModel EnglishMobile => new("English V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("en_PP-OCRv5_rec_mobile_infer.onnx", "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/en_PP-OCRv5_rec_mobile_infer.onnx"));

        public static OcrModel KoreanMobile => new("Korean V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("korean_PP-OCRv5_rec_mobile_infer.onnx", "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/korean_PP-OCRv5_rec_mobile_infer.onnx"));

        public static OcrModel LatinMobile => new("Latin V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("latin_PP-OCRv5_rec_mobile_infer.onnx", "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/latin_PP-OCRv5_rec_mobile_infer.onnx"));

        public static OcrModel ArabicMobile => new("Arabic V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("arabic_PP-OCRv5_rec_mobile_infer.onnx", "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/arabic_PP-OCRv5_rec_mobile_infer.onnx"));

        public static OcrModel DevanagariMobile => new("Devanagari V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("devanagari_PP-OCRv5_rec_mobile_infer.onnx", "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/devanagari_PP-OCRv5_rec_mobile_infer.onnx"));
        public static OcrModel TamilMobile => new("Tamil V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("ta_PP-OCRv5_rec_mobile_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ta_PP-OCRv5_rec_mobile_infer.onnx"));

        public static OcrModel TeluguMobile => new("Telugu V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("te_PP-OCRv5_rec_mobile_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/te_PP-OCRv5_rec_mobile_infer.onnx"));

        public static OcrModel ThaiMobile => new("Thai V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("th_PP-OCRv5_rec_mobile_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/th_PP-OCRv5_rec_mobile_infer.onnx"));

        public static OcrModel GreekMobile => new("Greek V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("el_PP-OCRv5_rec_mobile_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/el_PP-OCRv5_rec_mobile_infer.onnx"));

        public static OcrModel CyrillicMobile => new("Cyrillic V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("cyrillic_PP-OCRv5_rec_mobile_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/cyrillic_PP-OCRv5_rec_mobile_infer.onnx"));

        public static OcrModel EastSlavicMobile => new("East Slavic V5 Mobile", ModelVersion.V5, DetV5Mobile, DefaultCls,
            new("eslav_PP-OCRv5_rec_mobile_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/eslav_PP-OCRv5_rec_mobile_infer.onnx"));
    }

    public static class V4
    {
        // ---- Detectors ----

        private static readonly ModelSource DetV4Mobile = new(
            "ch_PP-OCRv4_det_infer.onnx",
            "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ch_PP-OCRv4_det_infer.onnx"
        );

        private static readonly ModelSource DetV4Server = new(
            "ch_PP-OCRv4_det_server_infer.onnx",
            "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ch_PP-OCRv4_det_server_infer.onnx"
        );

        private static readonly ModelSource DetV4English = new(
            "en_PP-OCRv3_det_infer.onnx",
            "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/en_PP-OCRv3_det_infer.onnx"
        );

        private static readonly ModelSource DetV4Multi = new(
            "Multilingual_PP-OCRv3_det_infer.onnx",
            "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/Multilingual_PP-OCRv3_det_infer.onnx"
        );

        // ---- Chinese ----

        public static OcrModel ChineseMobile => new(
            "Chinese V4 Mobile",
            ModelVersion.V4,
            DetV4Mobile,
            DefaultCls,
            new(
                "ch_PP-OCRv4_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ch_PP-OCRv4_rec_infer.onnx"
            )
        );

        public static OcrModel ChineseServer => new(
            "Chinese V4 Server",
            ModelVersion.V4,
            DetV4Server,
            DefaultCls,
            new(
                "ch_PP-OCRv4_rec_server_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ch_PP-OCRv4_rec_server_infer.onnx"
            )
        );

        public static OcrModel ChineseDocumentServer => new(
            "Chinese Document V4 Server",
            ModelVersion.V4,
            DetV4Server,
            DefaultCls,
            new(
                "ch_doc_PP-OCRv4_rec_server_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ch_doc_PP-OCRv4_rec_server_infer.onnx"
            )
        );

        public static OcrModel ChineseTraditional => new(
            "Chinese Traditional V4",
            ModelVersion.V4,
            DetV4Mobile,
            DefaultCls,
            new(
                "chinese_cht_PP-OCRv3_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/chinese_cht_PP-OCRv3_rec_infer.onnx"
            )
        );

        // ---- English / Latin ----

        public static OcrModel EnglishMobile => new(
            "English V4 Mobile",
            ModelVersion.V4,
            DetV4English,
            DefaultCls,
            new(
                "en_PP-OCRv4_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/en_PP-OCRv4_rec_infer.onnx"
            )
        );

        public static OcrModel LatinMobile => new(
            "Latin V4 Mobile",
            ModelVersion.V4,
            DetV4Multi,
            DefaultCls,
            new(
                "latin_PP-OCRv3_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/latin_PP-OCRv3_rec_infer.onnx"
            )
        );

        // ---- CJK ----

        public static OcrModel JapaneseMobile => new(
            "Japanese V4 Mobile",
            ModelVersion.V4,
            DetV4Mobile,
            DefaultCls,
            new(
                "japan_PP-OCRv4_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/japan_PP-OCRv4_rec_infer.onnx"
            )
        );

        public static OcrModel KoreanMobile => new(
            "Korean V4 Mobile",
            ModelVersion.V4,
            DetV4Mobile,
            DefaultCls,
            new(
                "korean_PP-OCRv4_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/korean_PP-OCRv4_rec_infer.onnx"
            )
        );

        // ---- Indic scripts ----

        public static OcrModel DevanagariMobile => new(
            "Devanagari V4 Mobile",
            ModelVersion.V4,
            DetV4Multi,
            DefaultCls,
            new(
                "devanagari_PP-OCRv4_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/devanagari_PP-OCRv4_rec_infer.onnx"
            )
        );

        public static OcrModel TamilMobile => new(
            "Tamil V4 Mobile",
            ModelVersion.V4,
            DetV4Multi,
            DefaultCls,
            new(
                "ta_PP-OCRv4_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ta_PP-OCRv4_rec_infer.onnx"
            )
        );

        public static OcrModel TeluguMobile => new(
            "Telugu V4 Mobile",
            ModelVersion.V4,
            DetV4Multi,
            DefaultCls,
            new(
                "te_PP-OCRv4_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/te_PP-OCRv4_rec_infer.onnx"
            )
        );

        public static OcrModel KannadaMobile => new(
            "Kannada V4 Mobile",
            ModelVersion.V4,
            DetV4Multi,
            DefaultCls,
            new(
                "ka_PP-OCRv4_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/ka_PP-OCRv4_rec_infer.onnx"
            )
        );

        // ---- RTL / Slavic ----

        public static OcrModel ArabicMobile => new(
            "Arabic V4 Mobile",
            ModelVersion.V4,
            DetV4Multi,
            DefaultCls,
            new(
                "arabic_PP-OCRv4_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/arabic_PP-OCRv4_rec_infer.onnx"
            )
        );

        public static OcrModel CyrillicMobile => new(
            "Cyrillic V4 Mobile",
            ModelVersion.V4,
            DetV4Multi,
            DefaultCls,
            new(
                "cyrillic_PP-OCRv3_rec_infer.onnx",
                "https://github.com/SnapXL/RapidOcrNet-ImageSharp/releases/download/v3.5.0/cyrillic_PP-OCRv3_rec_infer.onnx"
            )
        );
    }
}