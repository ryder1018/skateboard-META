using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Lets the player choose the Muse Image art style the server paints the level in.
/// The choice is sent as the "style" upload field and remembered in PlayerPrefs.
/// When the scene has no row of style buttons, one is built under the upload panel.
/// </summary>
public class ArtStyleSelector : MonoBehaviour
{
    [System.Serializable]
    public class StyleOption
    {
        [Tooltip("Style name sent to the server (wsl-cv muse_image.STYLES). \"original\" keeps the photo's own pixels.")]
        public string name;
        public string label;
        public Color color = Color.white;
    }

    private const string PrefsKey = "SelectedArtStyle";
    private const string RowName = "ArtStyleRow";

    [SerializeField]
    private List<StyleOption> styles = new List<StyleOption>
    {
        new StyleOption { name = "original", label = "Original", color = new Color32(0xe8, 0xe8, 0xe8, 0xff) },
        new StyleOption { name = "comic", label = "Comic", color = new Color32(0xff, 0xd2, 0x3f, 0xff) },
        new StyleOption { name = "cyberpunk", label = "Cyberpunk", color = new Color32(0xff, 0x3f, 0xd8, 0xff) },
        new StyleOption { name = "snow", label = "Snow", color = new Color32(0x9f, 0xe3, 0xff, 0xff) },
        new StyleOption { name = "pixel", label = "Pixel", color = new Color32(0x7c, 0xff, 0x6b, 0xff) },
        new StyleOption { name = "watercolor", label = "Watercolor", color = new Color32(0xff, 0x9a, 0x6b, 0xff) },
    };

    [Header("Generated Layout (1920x1080 reference)")]
    [SerializeField] private Vector2 rowPosition = new Vector2(0f, -400f);
    [SerializeField] private Vector2 chipSize = new Vector2(230f, 84f);
    [SerializeField] private float chipSpacing = 18f;

    private static readonly Color ChipIdleColor = new Color(0.07f, 0.08f, 0.1f, 0.88f);
    private static readonly Color ChipSelectedTextColor = new Color(0.07f, 0.08f, 0.1f, 1f);

    private readonly List<Image> chipImages = new List<Image>();
    private readonly List<TMP_Text> chipTexts = new List<TMP_Text>();
    private int selectedIndex = -1;

    /// <summary>Server style name of the current choice ("original" = the photo's pixels).</summary>
    public string SelectedStyle => SelectedOption != null ? SelectedOption.name : "original";

    /// <summary>Display label of the current choice.</summary>
    public string SelectedLabel => SelectedOption != null ? SelectedOption.label : "Original";

    private StyleOption SelectedOption
    {
        get
        {
            EnsureSelectionLoaded();
            return selectedIndex >= 0 && selectedIndex < styles.Count ? styles[selectedIndex] : null;
        }
    }

    private void Awake()
    {
        EnsureSelectionLoaded();
        BuildRow();
        RefreshChips();
    }

    /// <summary>Select a style by its server name; unknown names are ignored.</summary>
    public void Select(string styleName)
    {
        int index = styles.FindIndex(option => option.name == styleName);
        if (index >= 0)
        {
            SelectIndex(index);
        }
    }

    private void SelectIndex(int index)
    {
        selectedIndex = index;
        PlayerPrefs.SetString(PrefsKey, styles[index].name);
        PlayerPrefs.Save();
        RefreshChips();
        Debug.Log($"[ArtStyleSelector] Art style: {styles[index].label}");
    }

    private void EnsureSelectionLoaded()
    {
        if (selectedIndex >= 0 || styles.Count == 0)
        {
            return;
        }

        string saved = PlayerPrefs.GetString(PrefsKey, styles[0].name);
        selectedIndex = Mathf.Max(0, styles.FindIndex(option => option.name == saved));
    }

    private void BuildRow()
    {
        if (transform.Find(RowName) != null || styles.Count == 0)
        {
            return;
        }

        GameObject row = CreateUIObject(RowName, transform);
        RectTransform rowRect = row.GetComponent<RectTransform>();
        rowRect.anchorMin = new Vector2(0.5f, 0.5f);
        rowRect.anchorMax = new Vector2(0.5f, 0.5f);
        rowRect.pivot = new Vector2(0.5f, 0.5f);
        rowRect.anchoredPosition = rowPosition;
        rowRect.sizeDelta = new Vector2(styles.Count * chipSize.x + (styles.Count - 1) * chipSpacing, chipSize.y + 56f);

        TMP_Text title = CreateText("Title", row.transform, 30f, FontStyles.Bold);
        title.text = "MAP STYLE  (Muse Image)";
        title.color = Color.white;
        RectTransform titleRect = title.rectTransform;
        titleRect.anchorMin = new Vector2(0f, 1f);
        titleRect.anchorMax = new Vector2(1f, 1f);
        titleRect.pivot = new Vector2(0.5f, 1f);
        titleRect.anchoredPosition = Vector2.zero;
        titleRect.sizeDelta = new Vector2(0f, 44f);

        GameObject chips = CreateUIObject("Chips", row.transform);
        RectTransform chipsRect = chips.GetComponent<RectTransform>();
        chipsRect.anchorMin = new Vector2(0f, 0f);
        chipsRect.anchorMax = new Vector2(1f, 0f);
        chipsRect.pivot = new Vector2(0.5f, 0f);
        chipsRect.anchoredPosition = Vector2.zero;
        chipsRect.sizeDelta = new Vector2(0f, chipSize.y);

        HorizontalLayoutGroup layout = chips.AddComponent<HorizontalLayoutGroup>();
        layout.spacing = chipSpacing;
        layout.childAlignment = TextAnchor.MiddleCenter;
        layout.childControlWidth = true;
        layout.childControlHeight = true;
        layout.childForceExpandWidth = true;
        layout.childForceExpandHeight = true;

        for (int i = 0; i < styles.Count; i++)
        {
            int index = i;
            GameObject chip = CreateUIObject($"Style_{styles[i].name}", chips.transform);
            Image image = chip.AddComponent<Image>();
            Button button = chip.AddComponent<Button>();
            button.targetGraphic = image;
            button.onClick.AddListener(() => SelectIndex(index));

            TMP_Text text = CreateText("Text", chip.transform, 28f, FontStyles.Bold);
            text.text = styles[i].label;
            RectTransform textRect = text.rectTransform;
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = Vector2.zero;
            textRect.offsetMax = Vector2.zero;

            chipImages.Add(image);
            chipTexts.Add(text);
        }
    }

    private void RefreshChips()
    {
        for (int i = 0; i < chipImages.Count && i < styles.Count; i++)
        {
            bool selected = i == selectedIndex;
            chipImages[i].color = selected ? styles[i].color : ChipIdleColor;
            chipTexts[i].color = selected ? ChipSelectedTextColor : styles[i].color;
            chipImages[i].transform.localScale = selected ? new Vector3(1.06f, 1.06f, 1f) : Vector3.one;
        }
    }

    private static GameObject CreateUIObject(string objectName, Transform parent)
    {
        GameObject obj = new GameObject(objectName, typeof(RectTransform), typeof(CanvasRenderer));
        obj.transform.SetParent(parent, false);
        return obj;
    }

    private static TMP_Text CreateText(string objectName, Transform parent, float fontSize, FontStyles fontStyle)
    {
        GameObject textObject = CreateUIObject(objectName, parent);
        TextMeshProUGUI text = textObject.AddComponent<TextMeshProUGUI>();
        text.alignment = TextAlignmentOptions.Center;
        text.fontSize = fontSize;
        text.fontStyle = fontStyle;
        text.raycastTarget = false;
        return text;
    }
}
