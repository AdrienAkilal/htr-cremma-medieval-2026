"""
tests/test_alto_parser.py
--------------------------
Tests unitaires du parser ALTO XML.
"""
import pytest
from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.segmentation.alto_parser import parse_alto, get_image_filename


ALTO_V4_SAMPLE = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Description>
    <sourceImageInformation>
      <fileName>page001.jpg</fileName>
    </sourceImageInformation>
  </Description>
  <Layout>
    <Page WIDTH="1000" HEIGHT="1400">
      <PrintSpace>
        <TextBlock ID="b_001">
          <TextLine ID="l_001" HPOS="10" VPOS="20" WIDTH="500" HEIGHT="30">
            <String CONTENT="ce est li romans de la rose" HPOS="10" VPOS="20" WIDTH="480" HEIGHT="28"/>
          </TextLine>
          <TextLine ID="l_002" HPOS="10" VPOS="60" WIDTH="490" HEIGHT="30">
            <String CONTENT="n art damors est tote eclos" HPOS="10" VPOS="60" WIDTH="470" HEIGHT="28"/>
          </TextLine>
          <TextLine ID="l_003" HPOS="10" VPOS="100" WIDTH="100" HEIGHT="25">
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
""")


ALTO_V4_MULTISTRING = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout>
    <Page>
      <PrintSpace>
        <TextBlock>
          <TextLine ID="l_001" HPOS="5" VPOS="10" WIDTH="300" HEIGHT="20">
            <String CONTENT="bonjour" HPOS="5" VPOS="10" WIDTH="80" HEIGHT="20"/>
            <SP/>
            <String CONTENT="monde" HPOS="90" VPOS="10" WIDTH="60" HEIGHT="20"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
""")


@pytest.fixture
def alto_file(tmp_path):
    f = tmp_path / "sample.xml"
    f.write_text(ALTO_V4_SAMPLE, encoding="utf-8")
    return str(f)


@pytest.fixture
def alto_multistring_file(tmp_path):
    f = tmp_path / "multi.xml"
    f.write_text(ALTO_V4_MULTISTRING, encoding="utf-8")
    return str(f)


class TestParseAlto:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_alto("fichier_inexistant.xml")

    def test_returns_list(self, alto_file):
        lines = parse_alto(alto_file)
        assert isinstance(lines, list)

    def test_correct_line_count(self, alto_file):
        """L_003 est vide → seulement 2 lignes non vides."""
        lines = parse_alto(alto_file)
        assert len(lines) == 2

    def test_text_content(self, alto_file):
        lines = parse_alto(alto_file)
        assert lines[0]["text"] == "ce est li romans de la rose"
        assert lines[1]["text"] == "n art damors est tote eclos"

    def test_line_ids(self, alto_file):
        lines = parse_alto(alto_file)
        assert lines[0]["line_id"] == "l_001"
        assert lines[1]["line_id"] == "l_002"

    def test_polygon_shape(self, alto_file):
        lines = parse_alto(alto_file)
        poly = lines[0]["polygon"]
        assert len(poly) == 4
        for point in poly:
            assert len(point) == 2

    def test_polygon_coordinates(self, alto_file):
        lines = parse_alto(alto_file)
        # l_001 : HPOS=10, VPOS=20, WIDTH=500, HEIGHT=30
        poly = lines[0]["polygon"]
        assert poly[0] == [10, 20]           # haut-gauche
        assert poly[1] == [510, 20]          # haut-droit
        assert poly[2] == [510, 50]          # bas-droit
        assert poly[3] == [10, 50]           # bas-gauche

    def test_baseline_is_empty_list(self, alto_file):
        lines = parse_alto(alto_file)
        assert lines[0]["baseline"] == []

    def test_multistring_joined(self, alto_multistring_file):
        lines = parse_alto(alto_multistring_file)
        assert len(lines) == 1
        assert lines[0]["text"] == "bonjour monde"

    def test_invalid_xml_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.xml"
        bad.write_text("ceci n'est pas du XML <<<", encoding="utf-8")
        result = parse_alto(str(bad))
        assert result == []


class TestGetImageFilename:
    def test_reads_filename(self, alto_file):
        name = get_image_filename(alto_file)
        assert name == "page001.jpg"

    def test_missing_returns_none(self, tmp_path):
        f = tmp_path / "no_img.xml"
        f.write_text(
            '<?xml version="1.0"?><alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">'
            '<Layout/></alto>',
            encoding="utf-8"
        )
        assert get_image_filename(str(f)) is None
