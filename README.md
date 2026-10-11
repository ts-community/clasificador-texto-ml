# detector-busqueda-equipo

Clasificador de texto en español que detecta si un mensaje de Discord **busca compañeros para jugar a
Brawl Stars** o no. Sirve, por ejemplo, para que un bot mencione un rol de "buscar equipo" solo cuando
alguien lo pide de verdad.

| Mensaje | Resultado |
|---|---|
| `Busco gente para jugar ranked` | `looking` |
| `ola, ando push leg2` | `looking` |
| `Alguien ayuda a subir p3?` | `looking` |
| `Ya encontré equipo, gracias` | `not_looking` |
| `¿Qué brawler es mejor para ranked?` | `not_looking` |
| `Ayer llegué a leg2` | `not_looking` |

## Cómo funciona

- Modelo clásico de machine learning: **TF-IDF + regresión logística** (scikit-learn). No es una red neuronal
  ni un LLM. Entrena en segundos, sin GPU.
- Se entrena en Python (`train.py`) y se **exporta a un JSON** (`model/model.json`, ~0,2 MB).
- `typescript/detector.ts` lee ese JSON y puntúa mensajes en TypeScript puro: sin dependencias, sin Python
  ni servidor aparte, sin APIs de pago, ~0,05 ms por mensaje.
- Para no avisar de más, el umbral se elige priorizando la **precisión**: ante la duda, no detecta.

## Estructura

```
train.py                  entrena, evalúa y exporta el modelo
data/synthetic.csv        ejemplos etiquetados (text, label, group)
model/model.json          modelo ya entrenado (lo que usa el detector)
typescript/detector.ts    inferencia en TypeScript, para copiar a tu proyecto
requirements.txt          dependencias de Python
```

## Entrenar

Necesitas Python 3.11 o superior.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

Linux / macOS:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py
```

`train.py` imprime precisión y recall de `looking`, falsos positivos y falsos negativos, elige el umbral y
escribe `model/model.json`.

## Usar el modelo

Copia `typescript/detector.ts` y `model/model.json` a tu proyecto (Node 18+):

```ts
import { TeamSearchDetector } from "./detector.js";

const detector = TeamSearchDetector.fromFile("model/model.json"); // una sola vez al arrancar

const { score, decision } = detector.predict("Busco gente para jugar ranked");
// decision: "looking" | "not_looking"   ·   score: 0..1
// segundo argumento opcional para cambiar el umbral: detector.predict(texto, 0.95)
```

Ejemplo con discord.js, mencionando un rol solo si se detecta una búsqueda:

```ts
client.on(Events.MessageCreate, async (message) => {
  if (message.author.bot || message.channelId !== CHANNEL_ID) return;
  const text = message.content.trim();
  if (!text || text.length > 500) return;

  if (detector.predict(text).decision !== "looking") return;
  await message.reply({
    content: `<@&${ROLE_ID}>`,
    allowedMentions: { roles: [ROLE_ID], repliedUser: false },
  });
});
```

Recomendación: antes de activar las menciones, deja unos días el bot en modo observación (que solo registre
la decisión) y revisa los falsos positivos.

## Datos y etiquetado

`data/synthetic.csv` tiene las columnas `text`, `label` (`looking` | `not_looking`) y `group`. Contiene
mensajes con jerga, faltas, sin tildes y rangos del juego (`leg2`, `l1`, `p3`). `group` agrupa variantes de
una misma frase: `train.py` nunca las reparte entre entrenamiento y prueba, para que la evaluación no sea
engañosa.

Criterios usados:

- **Dudoso → `not_looking`** (`Alguien?`, `Vamos?`, `Ranked`): ante la duda, no se avisa.
- Ya encontró equipo, cancela o responde a otro (`me apunto`) → `not_looking`.
- Preguntas del juego, "busco" otra cosa (una skin, una guía) o buscar para otro juego → `not_looking`.
- Hablar del pasado (`ayer jugué ranked`) → `not_looking`.

Para ajustarlo a tu comunidad, crea `data/real.csv` con mensajes reales (columnas `text,label`), sin
menciones, IDs ni datos personales. `train.py` lo usa automáticamente y el archivo está en `.gitignore`.
Si cambias los datos, vuelve a ejecutar `train.py` y copia el nuevo `model/model.json`.

## Limitaciones

- Entiende palabras y trozos de palabra, no el significado: una forma de pedir equipo que no se parezca a
  ninguna de los datos puede no detectarse.
- Los datos incluidos son sintéticos, así que las métricas que imprime son orientativas. Valídalo con
  mensajes reales de tu servidor.
