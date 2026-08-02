"""裁剪模型：从多说话人模型中提取指定说话人，生成单说话人模型"""
import json
import sys

import torch

_load = torch.load
torch.load = lambda *a, **kw: _load(*a, **{**kw, 'weights_only': False})


def extract_speaker(model_path, config_path, speaker_name, out_model, out_config):
    ckpt = torch.load(model_path, map_location='cpu')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    spk_map = cfg.get('spk', {})
    if speaker_name not in spk_map:
        raise ValueError(f'说话人 {speaker_name} 不存在，可选: {list(spk_map.keys())}')

    spk_id = spk_map[speaker_name]
    n_speakers = len(spk_map)

    model_state = ckpt['model']
    emb_key = 'emb_g.weight'
    if emb_key in model_state:
        emb = model_state[emb_key]
        print(f'提取说话人 {speaker_name} (id={spk_id}), embedding shape: {emb.shape}')
        model_state[emb_key] = emb[spk_id:spk_id + 1].contiguous()
        ckpt['model'] = model_state

    cfg['spk'] = {speaker_name: 0}
    cfg['model']['n_speakers'] = 1

    torch.save(ckpt, out_model)
    with open(out_config, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f'模型已保存: {out_model}')
    print(f'配置已保存: {out_config}')
    print(f'spk: {cfg["spk"]}, n_speakers: 1')


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('用法: extract_speaker.py <model.pth> <config.json> <说话人名> <out_model.pth> <out_config.json>')
        sys.exit(1)
    extract_speaker(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
