# Clips del banco de pruebas

Cada clip son dos archivos con el mismo nombre:

| archivo | qué es |
|---|---|
| `<nombre>.wav` | el audio (mono, 16 kHz — cualquier formato si `faster-whisper` está instalado) |
| `<nombre>.json` | `{"language": "es", "text": "lo que se dice, palabra por palabra"}` |

Un `<nombre>.txt` con solo el texto también sirve; el `.json` gana si están los dos.

**El `text` es la vara.** Si está mal transcrito, el WER que reporta el banco miente
en la dirección más cara: te hace descartar una configuración que en realidad andaba
bien. Escribilo a mano, una vez, con cuidado.

## Clips incluidos

- **`librispeech-en-3081`** — 10,5s en inglés, de LibriSpeech test-clean
  (`3081-166546-0000`, CC BY 4.0, [openslr.org/12](https://www.openslr.org/12)).
  Está para que el banco corra apenas clonás el repo, sin grabar nada. **No alcanza
  para decidir nada sobre dictado real**: es inglés, leído, en estudio y limpio.

## Agregar clips propios (esto es lo que importa)

```bash
python benchmark/record_clip.py dictado-corto
python benchmark/record_clip.py dictado-largo --seconds 45
```

Graba desde tu micrófono, guarda el `.wav` y deja el `.json` con el `text` vacío
para que lo completes.

Qué grabar, para que el número sirva:

- **Tu micrófono, tu ruido de fondo, tu forma de hablar.** El banco mide *tu*
  hardware; un clip de estudio no predice nada sobre el tuyo.
- **Uno corto (~8s) y uno largo (~45s).** Es la comparación que decide todo: el
  one-shot de hoy tarda proporcional a lo que hablaste, el streaming debería quedar
  plano. Con un solo clip corto esa diferencia no se ve.
- **Jerga, nombres propios y números** (`Nobi`, `LayerCake`, `inox 316`, `de 31 a 8
  piezas`). Ahí es donde Whisper falla y donde el vocabulario custom de la app tiene
  que notarse.
- **Una pausa en el medio del clip largo.** El VAD y el trim del buffer se comportan
  distinto con silencio adentro, y dictando de verdad uno hace pausas.

> Ojo con los números: Whisper puede escribir `31` donde vos dijiste "treinta y uno".
> Las dos son correctas y el WER cuenta la segunda como error. Si te importa, escribí
> el ground truth como el modelo lo formatea, o dejá los números afuera de los clips.

Los `.wav` propios **no están gitignoreados**: son la referencia fija del banco y
tienen que viajar con el repo, si no la medición deja de ser reproducible. Son chicos
(~30 KB/s). Grabá solo lo que vas a usar.
