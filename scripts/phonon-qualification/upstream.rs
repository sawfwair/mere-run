// Qualification harness for the pinned, unmodified Gradium library.
use anyhow::{Context, Result};
use ptts::flow_lm::{ReplayRng, StepInput};
use ptts::plan::EosPolicy;
use ptts::tts_model::{TTSConfig, TTSModel};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf};
use xn::{CpuDevice, Unquantized};
#[derive(Deserialize)] struct Prompt { id: String, text: String }
#[derive(Serialize)] struct Step { latent: Vec<f32>, eos: f32 }
#[derive(Serialize)] struct Chunk { text: String, tokens: Vec<u32>, tail_frames: usize }
#[derive(Serialize)] struct Row { id: String, samples: usize, sample_rate: usize, frames: usize, chunks: Vec<Chunk> }
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    anyhow::ensure!((args.len() == 5 || (args.len() == 6 && args[5] == "--plan-only")), "MODEL CORPUS OUT NOISE_JSON LANGUAGE [--plan-only]");
    xn::set_num_threads(4);
    let root = PathBuf::from(&args[0]); let out = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&out)?;
    let cfg: TTSConfig = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
    let weights = ptts::loader::load_weights::<Unquantized<f32, CpuDevice>>(&root.join("model.safetensors"), &CpuDevice)?;
    let tokenizer = ptts::tok::Tok::open(&root.join("tokenizer.json"))?;
    let model: TTSModel<Unquantized<f32, CpuDevice>> = TTSModel::load(&weights, Box::new(tokenizer), &cfg, &HashMap::new())?;
    let voice = ptts::loader::load_voice_emb(&root.join("default-voice.safetensors"), cfg.embeddings_ext.as_deref(), model.speaker_proj(), &CpuDevice)?;
    let noise: Vec<f32> = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let prompts: Vec<Prompt> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let normalize: ptts::preprocess::Normalize = args[4].parse()?;
    let tok = model.flow_lm.conditioner.tokenizer.as_deref().context("tokenizer")?;
    let mut rows = vec![];
    for prompt in prompts {
        let chunks = ptts::plan::chunks(tok, &prompt.text, normalize, 50, cfg.mimi.frame_rate)?;
        let chunks = ptts::plan::fit_or_error(chunks, 200, tok, cfg.mimi.frame_rate)?;
        if args.len() == 6 {
            rows.push(Row { id: prompt.id, samples: 0, sample_rate: model.sample_rate(), frames: 0,
                chunks: chunks.iter().map(|x| Chunk { text: x.text.clone(), tokens: x.tokens.clone(), tail_frames: x.frames_after_eos }).collect() });
            continue;
        }
        let budget = chunks.iter().map(|c| c.seq_budget()).max().context("chunks")?;
        let mut base = model.init_flow_lm_state(1, budget)?;
        model.prompt_audio(&mut base, &voice)?;
        let mut rng = ReplayRng::new(noise.clone())?; let mut samples = vec![]; let mut steps = vec![];
        for chunk in &chunks {
            let mut state = base.clone(); model.prompt_text(&mut state, &chunk.tokens)?;
            let mut decoder = model.init_mimi_state(1)?; let mut previous = None;
            let mut eos = EosPolicy::new(chunk.frames_after_eos); let mut completed = false;
            for _ in 0..chunk.frame_budget {
                let input = match previous.as_ref() { None => StepInput::Bos { batch: 1 }, Some(t) => StepInput::Latent(t) };
                let (latent, logit) = model.generate_step_parts(&mut state, input, &mut rng)?;
                let logit = logit.to_vec()?[0];
                steps.push(Step { latent: latent.to_vec()?, eos: logit });
                let audio = model.decode_latent(&latent, &mut decoder)?.to_vec()?; samples.extend(audio);
                previous = Some(latent);
                if eos.should_stop(model.eos_from_logit(&[logit])) { completed = true; break }
            }
            anyhow::ensure!(completed, "frame budget exceeded: {}", prompt.id);
        }
        let bytes: Vec<u8> = samples.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{}.f32", prompt.id)), bytes)?;
        ptts::wav::write_wav_file(out.join(format!("{}.wav", prompt.id)), &samples, model.sample_rate().try_into()?)?;
        std::fs::write(out.join(format!("{}.steps.json", prompt.id)), serde_json::to_vec(&steps)?)?;
        rows.push(Row { id: prompt.id, samples: samples.len(), sample_rate: model.sample_rate(), frames: steps.len(),
            chunks: chunks.iter().map(|x| Chunk { text: x.text.clone(), tokens: x.tokens.clone(), tail_frames: x.frames_after_eos }).collect() });
    }
    std::fs::write(out.join("reference.json"), serde_json::to_vec_pretty(&rows)?)?;
    Ok(())
}
