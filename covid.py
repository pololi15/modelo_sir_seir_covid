import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.stats import gamma as gamma_dist
from scipy.signal import find_peaks

# === Parámetros ===
EXCEL_PATH = "p1.xlsx"
SMOOTH_WINDOW = 7
N = 11_800_000

# === Funciones de lectura (tu código original, con normalización Potosí) ===
def normalize_potosi_cols(df):
    for col in list(df.columns):
        if "Potos" in col or "PotosÃ" in col or "Potosí" in col or "Potosi" in col:
            if col != "Potosí":
                df = df.rename(columns={col: "Potosí"})
    return df

def sheet_to_national_series(df):
    if df.shape[1] < 2:
        return pd.DataFrame(columns=["fecha", "nacional"])
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "fecha"}).copy()
    try:
        df["fecha"] = pd.to_datetime(df["fecha"])
    except Exception:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).copy()
    df = normalize_potosi_cols(df)
    dept_cols = [c for c in df.columns if c != "fecha"]
    if len(dept_cols) == 0:
        return pd.DataFrame(columns=["fecha", "nacional"])
    df[dept_cols] = df[dept_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["nacional"] = df[dept_cols].sum(axis=1)
    df = df.groupby("fecha", as_index=False)["nacional"].sum()
    return df[["fecha", "nacional"]]

def detect_and_convert_cumulative(series):
    s = pd.Series(series).fillna(0.0)
    diffs = s.diff().fillna(0.0)
    negative_fraction = (diffs < -1e-6).sum() / max(1, len(diffs))
    if negative_fraction <= 0.05 and (s.max() > s.median() * 1.1):
        inc = diffs.clip(lower=0.0).values
        if inc[0] == 0 and s.iloc[0] > 0:
            inc[0] = s.iloc[0]
        return inc, True
    else:
        return s.values.clip(min=0.0), False

def load_bolivia_from_all_sheets(path):
    xls = pd.read_excel(path, sheet_name=None)
    partials = []
    for name, df in xls.items():
        try:
            part = sheet_to_national_series(df)
            if part.shape[0] > 0:
                partials.append(part)
        except Exception as e:
            print(f"Advertencia: error leyendo hoja '{name}': {e}")
    if len(partials) == 0:
        raise ValueError("No se encontraron datos útiles en ninguna hoja del Excel.")
    all_df = pd.concat(partials, axis=0, ignore_index=True)
    all_df = all_df.groupby("fecha", as_index=False)["nacional"].sum()
    all_df = all_df.sort_values("fecha").reset_index(drop=True)
    values, was_cumulative = detect_and_convert_cumulative(all_df["nacional"].values)
    if was_cumulative:
        print("Nota: Se detectó que los datos estaban en forma acumulada. Se convirtieron a incidencia diaria.")
    t = np.arange(len(values), dtype=float)
    fechas = all_df["fecha"].values
    return t, values.astype(float), fechas

# === Cargar datos ===
t, y_raw, fechas = load_bolivia_from_all_sheets(EXCEL_PATH)
if SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
    y = pd.Series(y_raw).rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().values
else:
    y = y_raw.copy()

# === Ecuaciones SIR / SEIR ===
def sir_ode(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def seir_ode(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def simulate_sir(t_arr, beta, gamma, N, I0=50.0):
    S0 = max(N - I0, 0.0); R0 = 0.0
    y0 = [S0, I0, R0]
    sol = odeint(sir_ode, y0, t_arr, args=(beta, gamma, N))
    S, I, R = sol.T
    inc = beta * S * I / N
    return S, I, R, inc

def simulate_seir(t_arr, beta, sigma, gamma, N, E0=50.0, I0=50.0):
    S0 = max(N - E0 - I0, 0.0); R0 = 0.0
    y0 = [S0, E0, I0, R0]
    sol = odeint(seir_ode, y0, t_arr, args=(beta, sigma, gamma, N))
    S, E, I, R = sol.T
    inc = beta * S * I / N
    return S, E, I, R, inc

# === Retardo: kernel Gamma (convulsión) ===
def gamma_kernel(mean_days=5.0, sd_days=2.0, L=30):
    if mean_days <= 0: mean_days = 1.0
    if sd_days <= 0: sd_days = 1.0
    k = (mean_days / sd_days) ** 2
    theta = (sd_days ** 2) / mean_days
    xs = np.arange(L)
    ker = gamma_dist.pdf(xs + 0.5, a=k, scale=theta)
    ker = ker / ker.sum()
    return ker

def apply_gamma_delay(series, mean_days=5.0, sd_days=2.0):
    ker = gamma_kernel(mean_days, sd_days, L=min(40, len(series)))
    conv = np.convolve(series, ker, mode='full')[:len(series)]
    return conv

# === Ajuste por ventana/ola ===
def fit_sir_window(t_win, y_win, N):
    # parámetros a ajustar: beta, gamma, p, I0, tau_days, fN
    def f(x, beta, gamma, p, I0, tau_days, fN):
        N_eff = max(1.0, min(N, fN * N))
        _, _, _, inc = simulate_sir(t_win, beta, gamma, N_eff, I0=I0)
        yhat = p * apply_gamma_delay(inc, mean_days=max(0.1, tau_days), sd_days=max(1.0, tau_days/2))
        return yhat
    p0 = [0.3, 1/7, 0.2, 50.0, 6.0, 0.05]
    lb = [1e-6, 1/21, 1e-6, 1.0, 0.0, 0.001]
    ub = [2.0, 1/3, 2.0, 1e7, 14.0, 0.5]
    try:
        popt, _ = curve_fit(f, t_win, y_win, p0=p0, bounds=(lb, ub), maxfev=200000)
        return popt
    except Exception as e:
        print("fit_sir_window fallo:", e)
        return None

def fit_seir_window(t_win, y_win, N, sigma=1/5.2, gamma_fixed=1/7.0):
    # parámetros a ajustar: beta, p, E0, I0, tau_days, fN  (sigma,y gamma fijos para identificabilidad)
    def f(x, beta, p, E0, I0, tau_days, fN):
        N_eff = max(1.0, min(N, fN * N))
        _, _, _, _, inc = simulate_seir(t_win, beta, sigma, gamma_fixed, N_eff, E0=E0, I0=I0)
        yhat = p * apply_gamma_delay(inc, mean_days=max(0.1, tau_days), sd_days=max(1.0, tau_days/2))
        return yhat
    p0 = [0.3, 0.2, 50.0, 50.0, 6.0, 0.05]
    lb = [1e-6, 1e-6, 1.0, 1.0, 0.0, 0.001]
    ub = [2.0, 2.0, 1e7, 1e7, 14.0, 0.5]
    try:
        popt, _ = curve_fit(f, t_win, y_win, p0=p0, bounds=(lb, ub), maxfev=300000)
        return popt
    except Exception as e:
        print("fit_seir_window fallo:", e)
        return None

# === Detectar picos/olas en la serie suavizada ===
# Ajusta estos parámetros si detecta demasiado poco/mucho
min_prominence = max(50, np.max(y) * 0.08)
min_distance = 14  # días entre picos
peaks, props = find_peaks(y, prominence=min_prominence, distance=min_distance)

# Si no encuentra picos, fuerza al menos uno en el máximo
if len(peaks) == 0:
    peaks = np.array([int(np.argmax(y))])

# Construir ventanas alrededor de cada pico: busca inicio y fin por caída relativa
windows = []
pad = 28  # días a expandir a cada lado (ajusta si quieres ventanas más cortas)
for p in peaks:
    start = max(0, p - pad)
    end = min(len(y), p + pad + 1)
    windows.append((start, end))

# Evitar solapamientos: fusiona ventanas que se traslapan
merged = []
for s, e in sorted(windows):
    if not merged:
        merged.append([s, e])
    else:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
windows = [(s, e) for s, e in merged]

print("Ventanas detectadas (start_idx, end_idx):", windows)

# === Arrays para almacenar predicciones por ola y totales ===
yhat_s_total = np.zeros_like(y)
yhat_e_total = np.zeros_like(y)
metadata = []

# Ajuste por cada ventana
for i, (s_idx, e_idx) in enumerate(windows, 1):
    t_win = np.arange(e_idx - s_idx, dtype=float)
    y_win = y[s_idx:e_idx]
    fechas_win = fechas[s_idx:e_idx]

    if len(y_win) < 8:
        print(f"Ola {i}: ventana muy corta, saltando.")
        continue

    print(f"\nAjustando ola {i}: índices {s_idx}:{e_idx}, longitud {len(y_win)} días")

    # Ajuste SIR
    popt_s = fit_sir_window(t_win, y_win, N)
    if popt_s is not None:
        beta_s, gamma_s, p_s, I0_s, tau_s, fN_s = popt_s
        N_eff_s = max(1.0, fN_s * N)
        _, _, _, inc_s = simulate_sir(t_win, beta_s, gamma_s, N_eff_s, I0=I0_s)
        yhat_s = p_s * apply_gamma_delay(inc_s, mean_days=max(0.1, tau_s), sd_days=max(1.0, tau_s/2))
        yhat_s_total[s_idx:e_idx] += yhat_s  # sumamos contribución de la ola
        metadata.append(("SIR", i, beta_s, gamma_s, p_s, I0_s, tau_s, fN_s))
        print(f"  SIR OK: beta={beta_s:.4f}, gamma={gamma_s:.4f}, p={p_s:.3f}, I0={I0_s:.1f}, tau={tau_s:.1f}, fN={fN_s:.4f}")
    else:
        print("  SIR: ajuste fallido.")

    # Ajuste SEIR
    popt_e = fit_seir_window(t_win, y_win, N)
    if popt_e is not None:
        beta_e, p_e, E0_e, I0_e, tau_e, fN_e = popt_e
        N_eff_e = max(1.0, fN_e * N)
        _, _, _, _, inc_e = simulate_seir(t_win, beta_e, 1/5.2, 1/7.0, N_eff_e, E0=E0_e, I0=I0_e)
        yhat_e = p_e * apply_gamma_delay(inc_e, mean_days=max(0.1, tau_e), sd_days=max(1.0, tau_e/2))
        yhat_e_total[s_idx:e_idx] += yhat_e
        metadata.append(("SEIR", i, beta_e, 1/5.2, 1/7.0, p_e, E0_e, I0_e, tau_e, fN_e))
        print(f"  SEIR OK: beta={beta_e:.4f}, sigma={1/5.2:.4f}, gamma={1/7.0:.4f}, p={p_e:.3f}, E0={E0_e:.1f}, I0={I0_e:.1f}, tau={tau_e:.1f}, fN={fN_e:.4f}")
    else:
        print("  SEIR: ajuste fallido.")

# === Graficar resultados ===
plt.figure(figsize=(14,6))
plt.plot(fechas, y, label="Confirmados diarios (suavizados)", color="black", lw=1.6)
plt.plot(fechas, yhat_s_total, label="Suma SIR por olas (predicción)", lw=1.4)
plt.plot(fechas, yhat_e_total, label="Suma SEIR por olas (predicción)", lw=1.4)
plt.title("Bolivia: Confirmados diarios vs SIR/SEIR (ajuste por olas)")
plt.xlabel("Fecha"); plt.ylabel("Casos/día")
plt.legend(); plt.tight_layout(); plt.show()

# Graficar por ventanas (subplots)
n = len(windows)
cols = 2
rows = (n + 1) // cols
if n > 0:
    fig, axs = plt.subplots(rows, cols, figsize=(12, 4*rows), squeeze=False)
    for idx, (s_idx, e_idx) in enumerate(windows):
        r = idx // cols; c = idx % cols
        ax = axs[r][c]
        fechas_win = fechas[s_idx:e_idx]
        ax.plot(fechas_win, y[s_idx:e_idx], label="Datos")
        ax.plot(fechas_win, yhat_s_total[s_idx:e_idx], label="SIR (ola)", linestyle="--")
        ax.plot(fechas_win, yhat_e_total[s_idx:e_idx], label="SEIR (ola)", linestyle="-.")
        ax.set_title(f"Ola {idx+1} ({s_idx}:{e_idx})")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    # ocultar ejes sobrantes
    total_axes = rows*cols
    for extra in range(n, total_axes):
        r = extra // cols; c = extra % cols
        fig.delaxes(axs[r][c])
    plt.tight_layout()
    plt.show()

# === Imprimir resumen de parámetros encontrados ===
print("\nResumen de parámetros encontrados (resumen parcial):")
for block in metadata:
    print(block)

# === Finalmente, calcula R0 aproximado para SIR/SEIR si están disponibles ===
print("\nR0 aproximados (si disponibles en metadata):")
for item in metadata:
    if item[0] == "SIR":
        _, ola, beta_s, gamma_s, *_ = item
        try:
            R0 = beta_s / gamma_s
            print(f"SIR - Ola {ola}: R0 ≈ {R0:.2f}")
        except:
            pass
    if item[0] == "SEIR":
        _, ola, beta_e, sigma_e, gamma_e, *_ = item
        try:
            R0 = beta_e / gamma_e
            print(f"SEIR - Ola {ola}: R0 ≈ {R0:.2f}")
        except:
            pass

