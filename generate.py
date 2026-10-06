"""Generate answers with DESCAPE.

Run a benchmark:
  python generate.py --model meta-llama/Llama-3.1-8B-Instruct --probe llama \
      --dataset truthfulqa --output_path outputs/truthfulqa_llama.json

Answer a single question:
  python generate.py --model meta-llama/Llama-3.1-8B-Instruct --probe llama \
      --question "Which country in Northern Europe has the best scores on PISA since 2015?"

`--probe` is a released probe name (llama / mistral / qwen) or a path to a probe checkpoint.
"""

import argparse
import json
import os

from tqdm import tqdm

from descape.decoder import DescapeDecoder
from descape.tasks import TASKS, exact_match, token_f1


def build_prompt(decoder, task, record):
    messages = [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": task.user_prompt(record)},
    ]
    return decoder.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser(description="DESCAPE decoding")
    parser.add_argument('--model', type=str, required=True,
                        help='Hugging Face model id or local path of the base LLM')
    parser.add_argument('--probe', type=str, required=True,
                        help='released probe name (llama / mistral / qwen) or checkpoint path')
    parser.add_argument('--dataset', type=str, default='truthfulqa', choices=list(TASKS))
    parser.add_argument('--question', type=str, default=None,
                        help='answer this single question instead of running a benchmark')
    parser.add_argument('--data_path', type=str, default=None,
                        help='FreshQA csv (freshqa) or FActScore prompt_entities.txt (factscore)')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--log_path', type=str, default=None,
                        help='write a detailed per-step decoding log to this file')
    parser.add_argument('--num_sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)

    # Beam search
    parser.add_argument('--beam_width', type=int, default=5,
                        help='Number of beams B')
    parser.add_argument('--candidates_per_beam', type=int, default=12,
                        help='Top-K candidates evaluated per beam')

    # Signal-integrated scoring
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Risk penalty weight')
    parser.add_argument('--tau', type=float, default=3.0,
                        help='Risk threshold')
    parser.add_argument('--gamma', type=float, default=0.3,
                        help='Factual bonus weight')
    parser.add_argument('--tau_fact', type=float, default=0.5,
                        help='Lower bound of the factual zone')

    # Generation
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum number of generated tokens (default: per benchmark)')
    parser.add_argument('--min_length', type=int, default=None,
                        help='Minimum number of generated tokens (default: per benchmark)')
    parser.add_argument('--repetition_penalty', type=float, default=1.2)
    parser.add_argument('--length_penalty', type=float, default=0.6,
                        help='Length normalization exponent')

    args = parser.parse_args()

    task = TASKS[args.dataset]
    if args.max_length is None:
        args.max_length = task.max_length
    if args.min_length is None:
        args.min_length = task.min_length
    if args.num_sample is None:
        args.num_sample = task.num_sample

    decoder = DescapeDecoder(
        model_path=args.model,
        probe_ckpt=args.probe,
        beam_width=args.beam_width,
        candidates_per_beam=args.candidates_per_beam,
        alpha=args.alpha,
        tau=args.tau,
        gamma=args.gamma,
        factual_lo=args.tau_fact,
        max_length=args.max_length,
        min_length=args.min_length,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        penalize_prompt_tokens=task.penalize_prompt_tokens,
    )
    if args.log_path:
        decoder.set_log_file(args.log_path)

    # ── Single question ───────────────────────────────────────────────
    if args.question is not None:
        key = 'entity' if args.dataset == 'factscore' else 'question'
        answer, _ = decoder.generate(build_prompt(decoder, task, {key: args.question}))
        decoder.close_log_file()
        print(f"\n{answer}")
        return

    # ── Benchmark ─────────────────────────────────────────────────────
    if not args.output_path:
        parser.error("--output_path is required when running a benchmark")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or '.',
                exist_ok=True)

    print(f"\nLoading {args.dataset} ...")
    records = task.load(args)
    print(f"  {len(records)} samples")

    results = []
    total_em, total_f1 = 0.0, 0.0
    pbar = tqdm(records, desc="Inference")

    for idx, record in enumerate(pbar):
        prompt = build_prompt(decoder, task, record)

        decoder._log(f"\n{'='*90}\nSAMPLE {idx}\n{'='*90}\n"
                     f"{record.get('question', record.get('entity'))}\n")
        answer, meta = decoder.generate(prompt, pbar=pbar)
        decoder._log(f"Prediction: {answer}")
        decoder._log("=" * 90)

        result = {'idx': idx, **record}
        if args.dataset == 'factscore':
            result.update({'topic': record['entity'], 'output': answer})
        result['prediction'] = answer
        if task.answer_key:
            result['em'] = exact_match(answer, record[task.answer_key])
            result['f1'] = token_f1(answer, record[task.answer_key])
            total_em += result['em']
            total_f1 += result['f1']
        result.update({
            'finished_beams': meta['finished_beams'],
            'dedup_count': meta['dedup_count'],
            'latency': meta['latency'],
        })
        results.append(result)

        total_tok = max(decoder.stats['total_tokens'], 1)
        total_time = max(decoder.stats['total_time'], 1e-9)
        postfix = f"Lat={total_time/total_tok*1000:.1f}ms/tok"
        if task.answer_key:
            n = idx + 1
            postfix = f"EM={total_em/n*100:.1f}%  F1={total_f1/n*100:.1f}%  " + postfix
        pbar.set_postfix_str(postfix)

    decoder.close_log_file()

    output = {
        'results': results,
        'statistics': decoder.stats,
        'config': {
            'dataset': args.dataset,
            'model': args.model,
            'probe': args.probe,
            'beam_width': args.beam_width,
            'candidates_per_beam': args.candidates_per_beam,
            'alpha': args.alpha,
            'tau': args.tau,
            'gamma': args.gamma,
            'tau_fact': args.tau_fact,
            'max_length': args.max_length,
            'min_length': args.min_length,
            'repetition_penalty': args.repetition_penalty,
            'length_penalty': args.length_penalty,
            'seed': args.seed,
        },
    }
    n = max(len(results), 1)
    if task.answer_key:
        output['metrics'] = {
            'exact_match': round(total_em / n * 100, 4),
            'token_f1': round(total_f1 / n * 100, 4),
            'n_samples': len(results),
        }

    print(f"\nSaving -> {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if args.dataset == 'factscore':
        # FActScore-compatible input file
        jsonl_path = os.path.splitext(args.output_path)[0] + '.jsonl'
        print(f"Saving JSONL -> {jsonl_path}")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps({'topic': r['topic'], 'output': r['output']},
                                   ensure_ascii=False) + '\n')

    total_tok = max(decoder.stats['total_tokens'], 1)
    total_time = max(decoder.stats['total_time'], 1e-9)

    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    if task.answer_key:
        print(f"  Exact Match (EM)      : {total_em / n * 100:.2f}%")
        print(f"  Token F1              : {total_f1 / n * 100:.2f}%")
    print(f"  Samples               : {len(results)}")
    print(f"  Total Tokens          : {total_tok}")
    print(f"  Avg Latency           : {total_time/total_tok*1000:.2f} ms/token")
    print(f"  Avg Throughput        : {total_tok/total_time:.2f} tokens/sec")
    print(f"  Early stops           : {decoder.stats['early_stops']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
