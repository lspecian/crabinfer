//! Tests for GPTQ/AWQ config parsing and weight loading.
//!
//! These tests validate the behavior contract for GPTQ and AWQ
//! quantize_config.json parsing, implemented in Plan 01-02.

#[cfg(test)]
mod tests {
    use crate::serving::safetensors_loader::{detect_quant_config, GptqConfig, AwqConfig};
    use crate::serving::quantization::QuantizationMethod;

    /// Verify GPTQ quantize_config.json deserializes with bits=4, group_size=128.
    #[test]
    fn test_gptq_config_parse() {
        let json = r#"{"bits": 4, "group_size": 128, "desc_act": false, "quant_method": "gptq"}"#;
        let cfg: GptqConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 128);
        assert!(!cfg.desc_act);
    }

    /// Verify AWQ config deserializes correctly with q_group_size alias.
    #[test]
    fn test_awq_config_parse() {
        let json = r#"{"quant_method": "awq", "q_group_size": 128, "w_bit": 4}"#;
        let cfg: AwqConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 128);
    }

    /// Verify that desc_act=true config is detected and rejected with an error.
    #[test]
    fn test_desc_act_true_rejected() {
        let tmp = std::env::temp_dir().join("crabinfer_test_stub_desc_act");
        std::fs::create_dir_all(&tmp).unwrap();

        let qcfg = r#"{"bits": 4, "group_size": 128, "desc_act": true, "quant_method": "gptq"}"#;
        std::fs::write(tmp.join("quantize_config.json"), qcfg).unwrap();
        std::fs::write(
            tmp.join("config.json"),
            r#"{"hidden_size": 4096, "intermediate_size": 11008, "num_attention_heads": 32, "num_hidden_layers": 32, "vocab_size": 32000}"#,
        ).unwrap();

        let result = detect_quant_config(&tmp);
        assert!(result.is_err(), "desc_act=true should produce an error");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("desc_act"),
            "error should mention desc_act, got: {err_msg}"
        );

        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
