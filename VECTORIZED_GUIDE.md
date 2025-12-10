# Vectorized Training - Quick Guide

## Czym jest vectorized training?

**Vectorized training** = wiele gier jednocześnie równolegle!

Zamiast trenować 1 grę → poczekać → następna gra, możesz trenować **4-8 gier jednocześnie** = **4-8x szybciej**!

---

## Podstawowe użycie

### Standardowy trening (1 gra naraz)
```bash
python train.py --headless --episodes 1000
```

### Vectorized - 4 gry równolegle (4x szybciej!)
```bash
python train.py --headless --episodes 1000 --num-envs 4
```

### Vectorized - 8 gier równolegle (8x szybciej!)
```bash
python train.py --headless --episodes 1000 --num-envs 8
```

---

## Parametry

### `--num-envs N`
Liczba równoległych środowisk (gier)

**Rekomendowane wartości:**
- `1` - standardowy (bez wektoryzacji)
- `4` - dobry balans (4x szybciej)
- `8` - bardzo szybko (8x szybciej)
- `16` - maksymalna prędkość (wymaga dużo RAM)

**Przykład:**
```bash
# 4 gry równolegle
python train.py --headless --episodes 1000 --num-envs 4
```

### `--async-envs`
Użyj asynchronicznej wektoryzacji (jeszcze szybsze dla wielu środowisk)

**Kiedy używać:**
- `--num-envs >= 8` - async jest szybsze
- `--num-envs < 8` - sync wystarczy

**Przykład:**
```bash
# 16 gier async (super szybko!)
python train.py --headless --episodes 1000 --num-envs 16 --async-envs
```

---

## Szybkie przykłady

### Test (super szybki)
```bash
# 4 gry x 200 epizodów = 800 gier w ~2-3 minuty
python train.py --headless --episodes 200 --num-envs 4
```

### Standardowy trening
```bash
# 4 gry x 1000 epizodów = 4000 gier w ~10-15 minut
python train.py --headless --episodes 1000 --num-envs 4
```

### Długi trening (najlepsze rezultaty)
```bash
# 8 gier x 2000 epizodów = 16000 gier w ~20-30 minut
python train.py --headless --episodes 2000 --num-envs 8
```

### Maksymalna prędkość
```bash
# 16 gier async x 2000 epizodów = 32000 gier w ~30-40 minut
python train.py --headless --episodes 2000 --num-envs 16 --async-envs
```

---

## Ile środowisk wybrać?

| num-envs | Prędkość | RAM | CPU | Kiedy używać |
|----------|----------|-----|-----|--------------|
| 1 | 1x (baseline) | Mało | Mało | Debug, rendering |
| 4 | ~4x | Średnio | Średnio | **Standardowe** |
| 8 | ~7x | Dużo | Dużo | Szybki trening |
| 16 | ~12x | Bardzo dużo | Bardzo dużo | Maksymalna prędkość |

**Rekomendacja dla M1 MacBook:**
- **4-8 envs** = optymalne
- 16+ może spowolnić przez zbyt dużą konkurencję

---

## Uwagi

### Rendering nie działa z vectorized
Nie możesz używać `--render-every` z `--num-envs > 1`.

**Rozwiązanie:**
1. Trenuj szybko z wektoryzacją
2. Potem użyj `play.py` żeby zobaczyć rezultat

### Episodes oznaczają "batche"
Przy `--num-envs 4`:
- `--episodes 100` = 100 batchy x 4 gry = **400 gier razem**
- `--episodes 1000` = 1000 batchy x 4 gry = **4000 gier razem**

### Log co ile sekund?
```bash
# Co ~10-15 sekund (przy 4 envs)
python train.py --headless --episodes 1000 --num-envs 4 --log-every 5

# Co ~30-60 sekund (przy 8 envs)
python train.py --headless --episodes 2000 --num-envs 8 --log-every 10
```

---

## Porównanie prędkości

**MacBook M1, WINNING_SCORE=5:**

| Konfiguracja | Czas na 1000 epizodów | Gier/sekunda |
|--------------|----------------------|--------------|
| 1 env | ~15 minut | ~1 eps/s |
| 4 envs | ~4 minuty | ~4 eps/s |
| 8 envs | ~2 minuty | ~8 eps/s |
| 16 envs async | ~1.5 minuty | ~11 eps/s |

---

## Typowy workflow

### 1. Szybki test (czy działa?)
```bash
python train.py --headless --episodes 100 --num-envs 4 --log-every 5
```
~1-2 minuty

### 2. Pełny trening
```bash
python train.py --headless --episodes 1000 --num-envs 8 --log-every 10
```
~10-15 minut

### 3. Zobacz rezultat
```bash
python play.py
```

---

## Troubleshooting

**"Out of memory":**
- Zmniejsz `--num-envs` (np. z 16 na 8)
- Zmniejsz `BATCH_SIZE` w config.py

**Trening nie szybszy:**
- Sprawdź użycie CPU: `top` lub Activity Monitor
- Jeśli CPU nie na 100%, zwiększ `--num-envs`
- Jeśli CPU na 100%, masz maksimum

**Wolniejsze niż 1 env:**
- Za dużo envs dla twojego CPU
- Zmniejsz `--num-envs`

---

## Stary kod (bez wektoryzacji) nadal działa!

Jeśli nie podasz `--num-envs`, kod działa jak wcześniej (1 gra naraz).

```bash
# Stary sposób - nadal działa
python train.py --headless --episodes 200
```
