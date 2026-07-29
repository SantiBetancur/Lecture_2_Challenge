"""Sistema de trazabilidad de transformaciones."""

import pandas as pd


class LogLimpieza:
    """Acumula cada transformación aplicada para exportarla como evidencia."""

    def __init__(self):
        self.entradas = []
        self._paso = 0

    def registrar(self, etapa, columna, accion, justificacion, afectados):
        self._paso += 1
        self.entradas.append(
            {
                "paso": self._paso,
                "etapa": etapa,
                "columna": columna,
                "accion": accion,
                "justificacion": justificacion,
                "registros_afectados": int(afectados),
            }
        )
        print(f"  [{self._paso:02d}] {columna:<22} {accion:<38} -> {afectados:>6} reg.")

    def to_frame(self):
        return pd.DataFrame(self.entradas)
