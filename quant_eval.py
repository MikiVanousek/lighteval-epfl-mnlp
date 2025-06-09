from pdb import run
import wandb
import yaml



import sys
import subprocess

def generate_cfg(model_id, dtype="none"):
    return f"""model:
  base_params:
    # Change <HF_USERNAME_team_member_QUANTIZED> to your own Huggingface username that holds the repo MNLP_M3_quantized_model
    # (Optional) If you want to use a chat template, set "use_chat_template=true" after revision.
    # (Optional) However, you must ensure that the chat template is saved in the model checkpoint.
    model_args: "pretrained={model_id},revision=main" 
    
    # If your model already has a quantization config as part of the model config, specify this as "none".
    # Otherwise. specify the model to be loaded in 4 bit. The other option is to use "8bit" quantization.
    dtype: "{dtype}"
    compile: false

  # Ignore this section, do not modify!
  merged_weights:
    delta_weights: false
    adapter_weights: false
    base_model: null
  generation:
    temperature: 0.0
"""


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_lighteval.py <path_to_yaml_config>")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = yaml.safe_load(open(config_path))
    model_name = cfg["hf_repo"]
    dtype = cfg.get("dtype", "none")
    

    wandb.init(
        project="nlp_quant",
        entity="vanousekmikulas-epfl",
        name=f"lighteval_{model_name}",
    )
    wandb.config.update({
        "model_name": model_name,
        "config_path": config_path,
    })


    cfg_text = generate_cfg(model_name, dtype=dtype)
    # create a temporary config file
    print(cfg_text)
    TMP_CONFIG_PATH = "temp_config.yaml"
    with open(TMP_CONFIG_PATH, "w") as f:
        f.write(cfg_text)


    command = [
        "lighteval", "accelerate",
        "--eval-mode", "lighteval",
        "--save-details",
        "--override-batch-size", "16",
        "--custom-tasks", "lighteval-epfl-mnlp/community_tasks/mnlp_mcqa_evals.py",
        "--output-dir", "../eval_output/",
        TMP_CONFIG_PATH,
        "community|mnlp_mcqa_evals|0|0"
    ]

    try:
        res = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        print(res.stdout.decode("utf-8") or "No output")

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode("utf-8") if e.stderr else "No stderr"
        stdout_output = e.stdout.decode("utf-8") if e.stdout else "No stdout"

        # Print for immediate debug
        print("LightEval execution failed:")
        print(f"stdout:\n{stdout_output}")
        print(f"stderr:\n{stderr_output}")

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
