# data/

Git ignores everything in this folder except this file. Do not commit photos, audio, catalogues or interaction logs.

## Synthetic data (default)

`sensehome generate --out data/synthetic` writes two files:

| File | Contents |
|---|---|
| `catalogue.json` | 240 synthetic items. Each item has a category, a title, a description, material and texture tags, and media references |
| `interactions.csv` | About 5,900 synthetic events (`view`, `save`, `purchase`) of 300 synthetic users, with timestamps |

The media references have the form `synthetic:color=sage;texture=woven;style=scandinavian;seed=12`
(photo) and `synthetic:material=oak;seed=12` (tap sound). The code renders the photo and the sound
from the reference in memory. No media file is written. If `data/synthetic` does not exist, the CLI
makes the same set in memory.

## Your own data

### `catalogue.json`

```json
{"items": [
  {"item_id": "sku-1001", "category": "sofa", "title": "Linen sofa", "description": "Three-seat sofa ...",
   "materials": ["linen", "oak"], "textures": ["woven"], "image": "images/sku-1001.jpg",
   "sound": "sounds/sku-1001.wav", "licence": "own photo"}
]}
```

- `category`: `sofa`, `armchair`, `table`, `chair`, `lamp`, `rug`, `shelf`, `bed`.
- `materials`: from `oak`, `walnut`, `pine`, `steel`, `brass`, `velvet`, `linen`, `leather`, `rattan`, `marble`, `wool`.
- `textures`: from `smooth`, `grainy`, `woven`, `ribbed`, `brushed`, `plush`, `rough`.
- `image` and `sound`: paths relative to the data folder, or `null`. A sound is optional. Use it only
  when the item has a real recorded sound (for example a tap on the surface in a product video).
- `licence`: the licence or the owner of the photo and the text. Do not use photos copied from shops.

### `interactions.csv`

Columns `user_id,item_id,timestamp,event`. `timestamp` is an integer (for example Unix seconds).
`event` is `view`, `save` or `purchase`. Each `item_id` must exist in the catalogue.

Install `pip install -e ".[media]"` to read real photos (Pillow) and WAV files (soundfile).

## Public sources that the prototype used, and why this project does not use them

- Product photos from online furniture shops: no licence for reuse.
- ESC-50 environmental sounds (<https://github.com/karolpiczak/ESC-50>, CC BY-NC 3.0): the sounds do not
  belong to the furniture items, and the licence is non-commercial.
- Describable Textures Dataset (<https://www.robots.ox.ac.uk/~vgg/data/dtd/>): texture photos that do not
  belong to the furniture items. They can be useful to pretrain a texture encoder. Check the terms first.
