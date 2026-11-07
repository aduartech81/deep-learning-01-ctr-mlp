**P1. ¿A partir de qué volumen PySpark supera a scikit-learn en tiempo sin degradar la calidad?**

En el rango medido **no se observó punto de cruce**: scikit-learn fue más rápido en todos los volúmenes en una sola máquina. **H1 se sostiene solo en su segunda parte** (por debajo del cruce scikit-learn es más eficiente); el cruce, si existe, está por encima del volumen probado o requiere un clúster real. La AUC de la carga fija fue 0.6683 (scikit-learn) vs 0.6660 (Spark) en promedio sobre los tamaños. Fracción secuencial de Amdahl: sklearn s = 1.00 (aceleración máxima teórica ≈ 1.0×); spark s = 0.21 (aceleración máxima teórica ≈ 4.9×).

**P2. ¿Cómo afecta la codificación de alta cardinalidad al desempeño y al costo?**

Con n = 10,000,000, MLP: freq: AUC 0.6717, log loss 0.4401, entrenamiento 184.0 s; hash: AUC 0.7422, log loss 0.3999, entrenamiento 1,337.4 s; target: AUC 0.7253, log loss 0.4086, entrenamiento 437.3 s. Memoria de la matriz de entrada: hash 2^12: 56 MB; hash 2^14: 56 MB; hash 2^16: 56 MB; freq: 27 MB; target: 27 MB. La diferencia de AUC hashing − frecuencia es +0.0705: **H2 no se sostiene tal como está formulada**; el hashing es claramente mejor, pero ocupa más memoria que la frecuencia (la hipótesis suponía lo contrario). Dimensión del hashing (ablación): 2^12: AUC 0.7269; 2^14: AUC 0.7295; 2^16: AUC 0.7276.

**P3. ¿Aporta el MLP una mejora significativa sobre líneas base más simples?**

- sk MLP freq vs spark MLP freq: ΔAUC = +0.0208 (p Holm = 0) → diferencia significativa
- sk MLP freq vs sk HGB freq: ΔAUC = -0.0534 (p Holm = 0) → diferencia significativa
- sk MLP freq vs sk SGD freq: ΔAUC = +0.0308 (p Holm = 0) → diferencia significativa
- spark MLP freq vs spark GBT freq: ΔAUC = -0.0528 (p Holm = 0) → diferencia significativa
- spark MLP freq vs spark LR freq: ΔAUC = +0.0157 (p Holm = 0) → diferencia significativa
- sk MLP hash vs sk HGB target: ΔAUC = +0.0031 (p Holm = 1.9e-100) → diferencia significativa, pero de magnitud pequeña (< 0,005)
- sk MLP hash vs sk SGD hash: ΔAUC = +0.0105 (p Holm = 0) → diferencia significativa
- sk MLP hash vs sk MLP freq: ΔAUC = +0.0705 (p Holm = 0) → diferencia significativa
- spark MLP hash vs spark LR hash: ΔAUC = +0.0083 (p Holm = 0) → diferencia significativa
- sk MLP hash (mismo n que Spark) vs spark MLP hash: ΔAUC = -0.0002 (p Holm = 0.016) → diferencia significativa, pero de magnitud pequeña (< 0,005)

**P4. ¿Son las explicaciones de LIME estables y fieles al modelo?**

Fidelidad media de LIME: R² = 0.118 (mínimo 0.069). Estabilidad entre semillas: Jaccard medio del top-5 = 0.777. Acuerdo con SHAP: Jaccard 0.595, Spearman 0.698. Coherencia global (Spearman de rankings): LIME–permutación 0.49, SHAP–permutación 0.48. Las explicaciones **no son suficientemente estables o fieles** para tomarlas como evidencia por sí solas: deben contrastarse con SHAP y con la importancia global.