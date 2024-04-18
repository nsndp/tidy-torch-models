def wconv_animesion(src, classify):
    wl = []
    for nm in src:
        if any(skip in nm for skip in ['text_embeddings', 'decoder', 'mlm_head']):
            continue
        if not classify and any(skip in nm for skip in ['model.fc', 'class_head.1']):
            continue
        elif 'positional_embedding' in nm:
            wl.insert(len(wl) - 2, (nm, src[nm]))
        elif 'norm1' in nm:
            wl.insert(len(wl) - 8, (nm, src[nm]))
        elif 'norm2' in nm:
            wl.insert(len(wl) - 4, (nm, src[nm]))
        else:
            wl.append((nm, src[nm]))
    return dict(wl)


def wconv_tv(src, classify):
    wl = []
    for nm in src:
        if nm == 'encoder.pos_embedding':
            wl.insert(len(wl) - 2, (nm, src[nm]))
        elif 'in_proj_weight' in nm:
                cb = nm.replace('_weight', '_bias')
                ws = src[nm].chunk(3)
                bs = src[cb].chunk(3)
                for i in range(3):
                    wl.append((nm + '_' + str(i), ws[i]))
                    wl.append((cb + '_' + str(i), bs[i]))
        elif 'in_proj_bias' in nm:
            continue
        elif not classify and 'heads.head' in nm:
            continue
        else:
            wl.append((nm, src[nm]))
    return dict(wl)


def wconv_openai(wd_clip):
    wl = []
    proj = None
    for nm in wd_clip:
        if nm.startswith('visual.'): # ignoring the text part, we only need ViT
            if 'attn.in_proj_weight' in nm:
                # proj_k/q/v are joined into one in OG CLIP dict, we need separate
                cb = nm.replace('_weight', '_bias')
                ws = wd_clip[nm].chunk(3)
                bs = wd_clip[cb].chunk(3)
                for i in range(3):
                    wl.append((nm + '_' + str(i), ws[i]))
                    wl.append((cb + '_' + str(i), bs[i]))
            elif 'attn.in_proj_bias' in nm:
                # skipping bias since we did it above alongside main weights
                continue
            elif 'ln_1' in nm:
                # norm 1 is after attention in OG CLIP dict, we need before
                wl.insert(len(wl) - 8, (nm, wd_clip[nm]))
            elif 'ln_2' in nm:
                # norm 2 is after feedforward in OG CLIP dict, we need before
                wl.insert(len(wl) - 4, (nm, wd_clip[nm]))
            elif 'class_embedding' in nm:
                wl.append((nm, wd_clip[nm].unsqueeze(0).unsqueeze(0)))
            elif 'positional_embedding' in nm:
                wl.append((nm, wd_clip[nm].unsqueeze(0)))
            elif nm == 'visual.proj':
                # projection weights are near the beginning, we need them last
                proj = (nm, wd_clip[nm].transpose(1, 0))
            else:
                wl.append((nm, wd_clip[nm]))
    wl.append(proj)
    return dict(wl)


def wconv_hface(wd_clip):
    wl = []
    for nm in wd_clip:
        if nm.startswith('vision_model.'):
            if 'position_ids' in nm:
                # just have numbers from 1 to 50, doesn't seem to be necessary as weights
                continue
            elif 'class_embedding' in nm:
                wl.append((nm, wd_clip[nm].unsqueeze(0).unsqueeze(0)))
            elif 'position_embedding' in nm:
                # placed after patch_embedding, we need before
                wl.insert(len(wl) - 1, (nm, wd_clip[nm].unsqueeze(0)))
            elif 'layer_norm1' in nm:
                # norm 1 is after attention, we need before
                wl.insert(len(wl) - 8, (nm, wd_clip[nm]))
            elif 'layer_norm2' in nm or 'q_proj' in nm:
                # norm 2 is after feedforward, we need before
                wl.insert(len(wl) - 4, (nm, wd_clip[nm]))
            elif 'q_proj' in nm:
                # this source have k, v, q; we need q, k, v
                wl.insert(len(wl) - 4, (nm, wd_clip[nm]))
            else:
                wl.append((nm, wd_clip[nm]))
    proj_name = 'visual_projection.weight'
    wl.append((proj_name, wd_clip[proj_name]))
    return dict(wl)


def wconv_facetr(src, classify):
    import torch
    wl = []
    for nm in src:
        if nm == 'cls_token':
            wl.insert(0, (nm, src[nm]))
        elif 'to_qkv' in nm:
            for i, ch in enumerate(src[nm].chunk(3)):
                wl.append((nm + '_' + str(i), ch))
                wl.append((nm.replace('.weight', '.bias') + '_' + str(i), torch.zeros(ch.shape[0])))
        elif not classify and nm == 'loss.weight':
            continue
        else:
            wl.append((nm, src[nm]))
    return dict(wl)