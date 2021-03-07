# Präfixsumme

# ImageFX Canny Kantenerkennung

## Setup
- Öffnen des ImageFX Ordners in Visual Studio (vorzugsweise VS 2019)
- Prüfen ob die OpenCL Libraries richting in der Projektkonfiguration verwiesen werden

Im Anschluss kann das Programm über den VS Debugger gestartet werden.

## Verwendung
Ist das Programm gestartet, gibt es drei Schalter die beschreibend bennant das tun was auf ihnen steht. 

Bei Druck auf _apply_ muss man darauf achten, dass in der Kommandozeile konfigurationen abgefragt werden, sobald diese eingegeben wurden startet die Berechnung.

Um ein anderes Bild zu wählen, muss man in Zeile 47 der `ImageFX.cpp` Datei auf dieses verweisen. Im _images_ Ordner liegen bereits einige Bilder zur verwenung bereit.

## Banane Beispiel

### Stage 0 Original
![](canny_examples/banana/Banane.png)
### Stage 1 Graustufen
![](canny_examples/banana/1.grey.bmp)
