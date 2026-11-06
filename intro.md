# Predicción de clics en anuncios móviles con MLP: scikit-learn vs PySpark

**Proyecto Integrador de Deep Learning (sección 2.12 de la guía del curso)**
Maestría en Ingeniería Biomédica · Universidad del Norte
Asignatura Deep Learning · Profesor: Dr. Lihki Rubio
Estudiante: Andrés Duarte Chaves

## Objetivo

Construir, evaluar y comparar clasificadores MLP que predicen si un usuario hace clic en un anuncio
móvil (`click = 1`) o no (`click = 0`), implementados en scikit-learn y en PySpark, bajo un protocolo
reproducible y libre de fuga de información, y analizar de forma crítica la interpretabilidad local
con LIME contrastándola con SHAP.

## Preguntas de investigación e hipótesis

| | Pregunta | Hipótesis |
|---|---|---|
| P1 | ¿A partir de qué volumen PySpark supera a scikit-learn en tiempo de entrenamiento sin degradar la calidad predictiva? | H1: existe un punto de cruce medible; por debajo de él scikit-learn es más eficiente. |
| P2 | ¿Cómo afecta la codificación de alta cardinalidad (hashing, frecuencia, target encoding) al desempeño y al costo? | H2: el hashing con dimensión moderada es comparable a la codificación por frecuencia con una fracción de la memoria. |
| P3 | ¿Aporta el MLP una mejora estadísticamente significativa sobre líneas base más simples? | Se contrasta con DeLong y McNemar. |
| P4 | ¿Son las explicaciones de LIME estables y fieles al modelo? | Se mide fidelidad (R² local) y estabilidad (Jaccard). |

## Decisiones de diseño

- **Datos:** Avazu completo (~40 M de impresiones, 21–30 de octubre de 2014). Se adopta la
  alternativa sin clúster que plantea el enunciado: P1 se trata como un experimento de punto de
  cruce en una sola máquina, sin presuponer que Spark gana.
- **Partición temporal:** entrenamiento días 21–28, validación día 29 y prueba día 30. La prueba
  nunca se usa para ajustar nada.
- **Anti-fuga:** vocabularios, umbral de raras, escaladores y codificadores se ajustan solo con
  entrenamiento; target encoding out-of-fold; umbral de decisión elegido en validación; parada
  temprana del MLP con el día 29 (no con una partición aleatoria).
- **Mismas filas en ambos entornos:** las muestras de 100 mil, 1 millón y 10 millones se
  materializan una vez (muestreo determinista por hash del `id`) y las leen scikit-learn y Spark.
- **Variabilidad:** 5 semillas en los tamaños pequeños y medianos (media, desviación e IC 95 %),
  más IC por bootstrap sobre la prueba.

## Estructura

1. Ficha de datos e ingesta
2. Análisis exploratorio
3. Preprocesamiento (misma lógica en ambos entornos)
4. Modelado con scikit-learn
5. Modelado con PySpark
6. Evaluación estadística
7. Benchmarking computacional
8. Interpretabilidad (LIME, SHAP, permutación)
9. Comparación y reflexión crítica

Anexo A: reproducibilidad.

El código reutilizable está en `src/ctr/` y toda la configuración experimental en `config.yaml`.
