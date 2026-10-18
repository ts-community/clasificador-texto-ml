# clasificador-texto-ml

Detector de intenciones para Discord con **NLP clásico**: le das etiquetas y devuelve un **porcentaje
por etiqueta**, local, offline y sin dependencias. Machine learning (**TF-IDF + regresión logística**)
entrenado en Python e inferido en TypeScript puro, a ~0,03 ms por mensaje. No es una red neuronal ni un
LLM.

Cada **módulo** es un clasificador independiente entrenado con tus ejemplos. Sirve, por ejemplo, para que
un bot de Discord mencione un rol solo cuando alguien pide algo de verdad.

Módulos incluidos:

| Módulo | Pregunta | Positivo | Negativo |
|---|---|---|---|
| `team_search` | ¿busca compañeros para jugar a Brawl Stars? | `looking` | `not_looking` |
| `participation` | ¿quiere participar en un minijuego/torneo? | `joining` | `not_joining` |

| Mensaje | `team_search` | `participation` |
|---|---|---|
| `Busco gente para jugar ranked` | `looking` | `not_joining` |
| `ola, ando push leg2` | `looking` | `not_joining` |
| `cuenta conmigo` | `not_looking` | `joining` |
| `me apunto al torneo` | `not_looking` | `joining` |
| `no puedo hoy` | `not_looking` | `not_joining` |
| `cuando es el torneo?` | `not_looking` | `not_joining` |

## Cómo funciona

- Modelo clásico de machine learning: **TF-IDF + regresión logística** (scikit-learn). Entrena en
  segundos, sin GPU.
- `train.py` entrena **todos** los datasets de `data/` y exporta un JSON por módulo (`model/<modulo>.json`).
- `typescript/detector.ts` lee esos JSON y puntúa mensajes en TypeScript puro: sin dependencias, sin
  Python ni servidor aparte, sin APIs de pago.
- Para no avisar de más, el umbral de cada módulo se elige priorizando la **precisión**: ante la duda,
  no detecta.
- Además, el score lleva un **suavizado L2** (`smooth`) y los datasets incluyen ejemplos de **temas
  ajenos** etiquetados como negativos (charla, dudas, "quiero comer", "me voy"...). Así una frase que no
  va del tema da un porcentaje muy bajo aunque contenga palabras comunes ("quiero", "voy"), y es muy
  difícil que dispare un falso positivo.

## Estructura

```
train.py                      entrena, evalúa y exporta un modelo por cada dataset de data/
data/team_search.csv          ejemplos etiquetados de team_search (text, label, group)
data/participation.csv        ejemplos etiquetados de participation
model/<modulo>.json           modelo entrenado + sus métricas (lo que usa el detector)
typescript/detector.ts        inferencia + DetectorSet, para copiar a tu proyecto
typescript/playground.ts      menú para probar frases (etiqueta, %, umbral, tiempo)
tools/python.mjs              lanzador de Python para los scripts de pnpm
requirements.txt              dependencias de Python
package.json / tsconfig.json  solo para el menú de TypeScript
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

Entrena todos los módulos; con `python train.py --module participation` entrena solo uno. Al terminar
imprime, por módulo, el tamaño del modelo y una tabla de calidad con porcentajes y errores:

```
╭────────────────────╮
│ Entrenando módulos │  2 módulo(s): participation, team_search
╰────────────────────╯

▍participation  (joining / not_joining)
  datos      1331 ejemplos · 263 grupos · joining 633 · not_joining 698 · modelo 0.10 MB · umbral 0.64
  métrica      validación       prueba
  precisión         95.0%        91.0%
  recall            90.7%        95.3%
  F1                92.8%        93.1%
  aciertos          93.3%        93.2%
  error abs            74           15
  error rel          6.7%         6.8%

▍team_search  (looking / not_looking)
  datos      2123 ejemplos · 259 grupos · looking 926 · not_looking 1197 · modelo 0.21 MB · umbral 0.66
  métrica      validación       prueba
  precisión         95.1%        91.8%
  ...

Resumen  (validación · verde = bien · amarillo = aceptable · rojo = a revisar)
  módulo         ejemplos  umbral  precisión  recall   error    tamaño
  participation      1331   0.640      95.0%   90.7%    6.7%    0.10 MB
  team_search        2123   0.660      95.1%   87.6%    7.4%    0.21 MB

✔ 2 módulo(s) exportado(s) -> model/
```

La salida usa color y negrita cuando se ejecuta en una terminal (se desactiva al redirigir la salida o
si defines `NO_COLOR`).

Si ya instalaste Node y ejecutaste `pnpm install`, el atajo `pnpm train` usa el Python del venv
automáticamente.

### Qué significa cada dato

| Dato | Significado |
|---|---|
| **probabilidad** | Confianza del modelo en la etiqueta positiva, de 0% a 100%. |
| **umbral** | Probabilidad mínima para marcar la etiqueta positiva. Si no llega, marca la negativa. |
| **precisión** | De lo que marcó como positivo, qué % acertó. Alta = pocas falsas alarmas. |
| **recall** | De los positivos reales, qué % encontró. Alto = se le escapan pocos casos. |
| **F1** | Media de precisión y recall en un solo número (0–100%). |
| **aciertos** | Porcentaje total de clasificaciones correctas. |
| **error abs** | Número de fallos (falsos positivos + falsos negativos). |
| **error rel** | Esos fallos como porcentaje del total. |

Y las dos columnas de evaluación (ambas **exigentes**: apartan plantillas enteras, es decir, obligan a
acertar frases cuya plantilla no se vio en el entrenamiento):

| Columna | Qué mide |
|---|---|
| **validación** | Validación cruzada por grupos (5 folds). Es la cifra de referencia. |
| **prueba** | Una única partición por grupos apartada, que no se toca hasta el final. |

Las métricas se colorean con un **semáforo** (igual en `train.py` y en el menú), para que al crear un
módulo nuevo veas de un vistazo por dónde falla:

| Métrica | Verde | Amarillo | Rojo |
|---|---|---|---|
| precisión | ≥ 92% | 85–92% | < 85% |
| recall | ≥ 85% | 70–85% | < 70% |
| F1 | ≥ 88% | 75–88% | < 75% |
| aciertos | ≥ 90% | 80–90% | < 80% |
| error rel | < 10% | 10–20% | ≥ 20% |

Sobre el umbral: se calcula **por módulo** para alcanzar una precisión objetivo (`--min-precision`, por
defecto 95%) con el mayor recall posible. Bajarlo sube recall y baja precisión; subirlo al revés. Y lo
más importante: para **mejorar de verdad** (precisión y recall a la vez) lo que hay que hacer es **añadir
más y mejores ejemplos**, no tocar el umbral.

## Probar frases (menú)

Requiere Node 24+ (ejecuta los `.ts` directamente, sin compilar). No hay dependencias en tiempo de
ejecución; `pnpm install` solo instala TypeScript para el typecheck opcional.

```powershell
node typescript/playground.ts          # menú interactivo
```

```
╭───────────────────────╮
│ Clasificador de texto │  2 módulo(s): participation, team_search · umbral: automático por módulo
╰───────────────────────╯

Calidad (validación) · tamaño
  módulo         ejemplos  umbral  precisión  recall   error    tamaño
  participation      1331   0.640      95.0%   90.7%    6.7%   0.10 MB
  team_search        2123   0.660      95.1%   87.6%    7.4%   0.21 MB
  verde = bien · amarillo = aceptable · rojo = a revisar  ·  detalle en :info

> me apunto al torneo
  participation  →  joining
    probabilidad  98.2%   umbral 64.5%   (supera)
    tiempo        0.24 ms
  team_search    →  not_looking
    probabilidad   1.5%   umbral 66.0%   (no llega)
    tiempo        0.20 ms
```

Cada frase se prueba contra todos los módulos y muestra la etiqueta, la probabilidad, el umbral y cuánto
tarda. Comandos:

- `:info [módulo]` — detalle del modelo (validación y prueba) y qué significa cada dato.
- `:use <a,b|all>` — elegir los módulos activos.
- `:thr <0..1|off>` — fijar un umbral manual para ver cómo cambia la decisión (marcado con `*`).
- `:stats` — resumen de la sesión (positivos y tiempo medio).
- `:help`, `:q`.

`:info` da el detalle por módulo, con la misma tabla que el entrenamiento:

```
participation  (joining / not_joining)
  datos     1331 ejemplos · participation.json · 0.10 MB · umbral 0.64 · entrenado 2026-10-18
  métrica      validación       prueba
  precisión         95.0%        91.0%
  recall            90.7%        95.3%
  F1                92.8%        93.1%
  aciertos          93.3%        93.2%
  error abs            74           15
  error rel          6.7%         6.8%
```

También sin menú:

```powershell
node typescript/playground.ts "me apunto al torneo"           # una frase contra todos los módulos
node typescript/playground.ts -m participation "me apunto"    # solo un módulo
node typescript/playground.ts -t 0.5 "puedo participar?"      # umbral manual
```

Atajos con pnpm:

```powershell
pnpm install                      # instala TypeScript y @types/node (solo para typecheck)
pnpm playground                   # menú interactivo
pnpm playground -- "me apunto"    # los argumentos tras -- van al menú
pnpm train                        # entrenar
pnpm typecheck                    # comprobar tipos
```

## Usar el modelo

Copia `typescript/detector.ts` y la carpeta `model/` a tu proyecto (Node 18+; no necesita los `.ts`
del menú):

```ts
import { DetectorSet } from "./detector.js";

const detectors = DetectorSet.fromDir("model"); // una sola vez al arrancar

const { score, decision } = detectors.predict("participation", "me apunto al torneo");
// decision: "joining" | "not_joining"   ·   score: 0..1 (el porcentaje)
// umbral manual opcional: detectors.predict("participation", texto, 0.95)

// porcentajes de todas las etiquetas/módulos de una vez:
// for (const r of detectors.predictAll(texto)) console.log(r.module, r.decision, r.score);
```

Ejemplo con discord.js, mencionando un rol solo si se detecta una búsqueda de equipo:

```ts
client.on(Events.MessageCreate, async (message) => {
  if (message.author.bot || message.channelId !== CHANNEL_ID) return;
  const text = message.content.trim();
  if (!text || text.length > 500) return;

  if (detectors.predict("team_search", text).decision !== "looking") return;
  await message.reply({
    content: `<@&${ROLE_ID}>`,
    allowedMentions: { roles: [ROLE_ID], repliedUser: false },
  });
});
```

Recomendación: antes de activar las menciones, deja unos días el bot en modo observación (que solo registre
la decisión) y revisa los falsos positivos.

## Crear un módulo nuevo

Un módulo es un fichero `data/<nombre>.csv` (columnas `text,label,group`):

1. Crea `data/<nombre>.csv` con ejemplos etiquetados. Debe haber **exactamente dos etiquetas** y la
   negativa debe empezar por `no_` o `not_` (p. ej. `joining` / `not_joining`).
2. Ejecuta `python train.py`. Se genera `model/<nombre>.json` (con sus métricas dentro).
3. Copia `typescript/detector.ts` y `model/` a tu bot. `DetectorSet` descubre el módulo solo.

No hace falta tocar código: `train.py` y `DetectorSet` recorren las carpetas `data/` y `model/`.

## Datos y etiquetado

Cada CSV tiene las columnas `text`, `label` (las dos etiquetas del módulo) y `group`. Contiene mensajes
con jerga, faltas, sin tildes y emojis. `group` agrupa variantes de una misma frase: `train.py` nunca las
reparte entre entrenamiento y prueba, para que la evaluación no sea engañosa.

Criterios usados:

- **Dudoso → etiqueta negativa** (`Alguien?`, `Ranked`, `si me da tiempo`): ante la duda, no se avisa.
- Ya resuelto, cancelado o respuesta a otro → negativo.
- Preguntas sobre la información (reglas, horarios, premios) o hablar de otro tema → negativo.
- Hablar del pasado (`ayer jugué ranked`) → negativo.

Para ajustarlo a tu comunidad, crea `data/<modulo>.real.csv` con mensajes reales (columnas `text,label`),
sin menciones, IDs ni datos personales. `train.py` lo usa automáticamente y el archivo está en
`.gitignore`. Si cambias los datos, vuelve a ejecutar `train.py` y copia los nuevos JSON.

## Limitaciones

- Entiende palabras y trozos de palabra, no el significado: una forma de pedir algo que no se parezca a
  ninguna de los datos puede no detectarse. Negaciones como `no participo` se aprenden por contexto léxico.
- Cada módulo necesita ejemplos etiquetados y su propio modelo; no clasifica etiquetas que no haya visto.
- Los datos incluidos son sintéticos, así que las métricas que imprime son orientativas. Valídalos con
  mensajes reales de tu servidor.
