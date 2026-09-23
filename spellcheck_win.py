"""
Corrector ortográfico del sistema, para MARCAR palabras que no existen.

POR QUÉ
-------
El modelo inventa palabras: MEDIDO en openwhisper.log salieron `superiferia`,
`internales`, `varija`, `fronten`, `retspicher`, `barch`. Eso, a diferencia de
un error entre dos palabras reales, se puede detectar solo: basta un
diccionario. No dice CUÁL era la correcta —para eso está corrections.py— pero
señala dónde mirar, que es la mitad del trabajo.

POR QUÉ EL DE WINDOWS Y NO UNA LIBRERÍA
---------------------------------------
MEDIDO sobre las 1290 palabras distintas del log del usuario:
  - pyspellchecker (es): marca el 53%. Da por inexistentes `está`, `quiero`,
    `cosas`, `tiene`, `vamos`. Es una lista plana de 86k palabras sin
    morfología, y el español conjuga demasiado. Inservible.
  - Windows Spell Checking API (es-AR): marca el 11%, y casi todo eso son
    palabras en inglés (`prompt`, `speech`, `power`), nombres propios sin
    mayúscula (`linkedin`, `spotify`) o tecleo de más (`eeeee`). 0,24 ms por
    frase.
Además no agrega dependencias (comtypes ya estaba), no empaqueta diccionarios
y no tiene licencia que revisar para embeberlo en nitoOS.

LÍMITE
------
Solo ve palabras que NO EXISTEN. `acerca` escrito `a cerca`, o `cooler` escrito
`culo`, son palabras reales y pasan derecho. Esto no reemplaza corregir a mano.

Uso:
    sc = SystemSpellChecker("es")
    sc.ignore_all(["nitoOS", "ReSpeaker"])   # el vocabulario propio no se marca
    sc.is_misspelled("superiferia")          # True
    sc.suggest("superiferia")                # ['periferia', 'soporifera', ...]
"""
from ctypes import POINTER, c_int, c_ulong, c_wchar_p

try:
    from comtypes import GUID, IUnknown, COMMETHOD, HRESULT, CoCreateInstance
    _COM_OK = True
except Exception:  # pragma: no cover - sin comtypes no hay corrector, no pasa nada
    _COM_OK = False

LPWSTR = c_wchar_p

# Las interfaces no están en ninguna typelib registrada, así que se declaran a
# mano. Los IID salen de spellcheck.idl del SDK de Windows; el orden de los
# métodos ES el de la vtable y no se puede tocar.
if _COM_OK:
    class IEnumString(IUnknown):
        _iid_ = GUID("{00000101-0000-0000-C000-000000000046}")
        _methods_ = [
            COMMETHOD([], HRESULT, "Next",
                      (["in"], c_ulong, "celt"),
                      (["out"], POINTER(LPWSTR), "rgelt"),
                      (["out"], POINTER(c_ulong), "pceltFetched")),
            COMMETHOD([], HRESULT, "Skip", (["in"], c_ulong, "celt")),
            COMMETHOD([], HRESULT, "Reset"),
            COMMETHOD([], HRESULT, "Clone",
                      (["out"], POINTER(POINTER(IUnknown)), "ppenum")),
        ]

    class ISpellingError(IUnknown):
        _iid_ = GUID("{B7C82D61-FBE8-4B47-9B27-6C0D2E0DE0A3}")
        _methods_ = [
            COMMETHOD([], HRESULT, "get_StartIndex", (["out"], POINTER(c_ulong), "v")),
            COMMETHOD([], HRESULT, "get_Length", (["out"], POINTER(c_ulong), "v")),
            COMMETHOD([], HRESULT, "get_CorrectiveAction", (["out"], POINTER(c_int), "v")),
            COMMETHOD([], HRESULT, "get_Replacement", (["out"], POINTER(LPWSTR), "v")),
        ]

    class IEnumSpellingError(IUnknown):
        _iid_ = GUID("{803E3BD4-2828-4410-8290-418D1D73C762}")
        _methods_ = [
            COMMETHOD([], HRESULT, "Next",
                      (["out"], POINTER(POINTER(ISpellingError)), "v")),
        ]

    class ISpellChecker(IUnknown):
        _iid_ = GUID("{B6FD0B71-E2BC-4653-8D05-F197E412770B}")
        _methods_ = [
            COMMETHOD([], HRESULT, "get_LanguageTag", (["out"], POINTER(LPWSTR), "v")),
            COMMETHOD([], HRESULT, "Check", (["in"], LPWSTR, "text"),
                      (["out"], POINTER(POINTER(IEnumSpellingError)), "v")),
            COMMETHOD([], HRESULT, "Suggest", (["in"], LPWSTR, "word"),
                      (["out"], POINTER(POINTER(IEnumString)), "v")),
            # Add() escribiría en el diccionario personal de Windows del
            # usuario: no se usa nunca. Ignore() vale solo para esta instancia,
            # que es exactamente lo que queremos para el vocabulario propio.
            COMMETHOD([], HRESULT, "Add", (["in"], LPWSTR, "word")),
            COMMETHOD([], HRESULT, "Ignore", (["in"], LPWSTR, "word")),
        ]

    class ISpellCheckerFactory(IUnknown):
        _iid_ = GUID("{8E018A9D-2415-4677-BF08-794EA61F94BB}")
        _methods_ = [
            COMMETHOD([], HRESULT, "get_SupportedLanguages",
                      (["out"], POINTER(POINTER(IEnumString)), "v")),
            COMMETHOD([], HRESULT, "IsSupported", (["in"], LPWSTR, "tag"),
                      (["out"], POINTER(c_int), "v")),
            COMMETHOD([], HRESULT, "CreateSpellChecker", (["in"], LPWSTR, "tag"),
                      (["out"], POINTER(POINTER(ISpellChecker)), "v")),
        ]

    CLSID_SpellCheckerFactory = GUID("{7AB36653-1796-484B-BDFA-E74F1DB7C1DC}")

# El usuario dicta en voseo rioplatense y Windows marca varias de esas formas
# (MEDIDO: `hacelo`, `tenés`, `guardame`, `decime`). Son correctas acá, así que
# se dan por buenas para que el subrayado no se vuelva ruido de fondo.
_VOSEO_SUFIJOS = ("ame", "ate", "eme", "ete", "ime", "ite", "alo", "elo", "ilo",
                  "ala", "ela", "ila", "anos", "enos", "inos", "és", "ás", "ís")


class SystemSpellChecker:
    """Envoltorio del corrector de Windows. Si algo falla queda inactivo y
    `is_misspelled` devuelve False siempre: la app anda igual, sin subrayados."""

    def __init__(self, language="es"):
        self._sc = None
        self._cache = {}
        if not _COM_OK:
            print("[Ortografia] comtypes no disponible; sin marcado.")
            return
        try:
            unk = CoCreateInstance(CLSID_SpellCheckerFactory, interface=IUnknown)
            factory = unk.QueryInterface(ISpellCheckerFactory)
            tag = self._pick_tag(factory, language)
            if tag is None:
                print(f"[Ortografia] Windows no tiene diccionario para '{language}'.")
                return
            self._sc = factory.CreateSpellChecker(tag)
            print(f"[Ortografia] Corrector del sistema: {tag}")
        except Exception as exc:  # noqa: BLE001 - nunca romper el dictado por esto
            print(f"[Ortografia] No pude abrir el corrector de Windows: {exc}")
            self._sc = None

    @staticmethod
    def _pick_tag(factory, language):
        """El idioma de la app es 'es'; Windows quiere una variante. Se prueba
        la rioplatense primero, después la genérica."""
        base = (language or "es").lower().split("-")[0]
        for tag in (f"{base}-AR", f"{base}-419", f"{base}-ES", base):
            try:
                if factory.IsSupported(tag):
                    return tag
            except Exception:
                continue
        return None

    @property
    def available(self):
        return self._sc is not None

    def ignore_all(self, words):
        """Sacar de sospecha el vocabulario propio del usuario (lo que escribió
        en Configuración y lo que fue corrigiendo). Vale solo para esta
        instancia: no toca el diccionario personal de Windows."""
        if self._sc is None:
            return
        for word in words:
            for part in str(word).split():
                part = part.strip()
                if not part:
                    continue
                try:
                    self._sc.Ignore(part)
                except Exception:
                    pass
                self._cache.pop(part.lower(), None)

    def is_misspelled(self, word):
        if self._sc is None or not word:
            return False
        key = word.lower()
        if key in self._cache:
            return self._cache[key]
        result = False
        if not self._is_voseo(key):
            try:
                result = self._has_error(self._sc.Check(word))
            except Exception:
                result = False
        self._cache[key] = result
        return result

    @staticmethod
    def _is_voseo(word):
        """Voseo rioplatense (`hacelo`, `decime`, `tenés`). Windows los marca y
        acá son correctos, así que ni se consultan."""
        return len(word) >= 5 and word.endswith(_VOSEO_SUFIJOS)

    @staticmethod
    def _has_error(enum):
        """¿El enumerador trae al menos un error?

        OJO: cuando no hay ninguno, `Next()` devuelve un puntero NULL, que en
        comtypes NO es None —es un POINTER falsy—, así que hay que preguntar
        por su verdad y no compararlo contra None. Con `is not None` daba TODA
        palabra por mal escrita, incluidas `está` y `quiero`.
        """
        try:
            return bool(enum.Next())
        except Exception:
            return False

    def suggest(self, word, limit=3):
        if self._sc is None or not word:
            return []
        out = []
        try:
            enum = self._sc.Suggest(word)
            while len(out) < limit:
                text, got = enum.Next(1)
                if not got:
                    break
                if text and text.lower() != word.lower():
                    out.append(text)
        except Exception:
            return []
        return out
