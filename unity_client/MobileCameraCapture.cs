using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Captures or selects photos, sends them to the server, polls the generated result,
/// and lets the result panel add or start the generated level.
/// Supports a batch flow for selecting multiple gallery images at once.
/// </summary>
public class MobileCameraCapture : MonoBehaviour
{
    [System.Serializable]
    public class ResponseImages
    {
        public string preview_base64;
        public string foreground_base64;
        public string gameplay_base64;
        public string background_base64;
    }

    [System.Serializable]
    public class ServerResponse
    {
        public string status;
        public string message;
        public ResponseImages images;
        public string foreground_base64;
        public string gameplay_base64;
        public string background_base64;
        public TrackData metadata;
    }

    [Header("Network Settings")]
    [Tooltip("Used when there is no saved IP in PlayerPrefs.")]
    public string defaultServerIP = "192.168.1.100";
    public string port = "5000";
    private string apiKey = "112550191";

    [Header("Polling Settings")]
    public int maxPollingAttempts = 5;
    public float pollingInterval = 1.0f;

    [Header("UI References")]
    public TMP_InputField ipInputField;
    public Button saveIPButton;

    [Header("Image Settings")]
    public int targetLongSide = 1920;
    [Range(1, 100)]
    public int jpgQuality = 85;

    [Header("Gallery Selection")]
    [Tooltip("When supported on the current mobile platform, the gallery button opens in multi-select mode.")]
    public bool preferMultipleGallerySelection = true;
    [Tooltip("If multi-select is unavailable, fall back to the original single-image picker.")]
    public bool fallbackToSingleGallerySelection = true;

    [Header("Architecture")]
    public LevelCoordinator levelCoordinator;
    [SerializeField] private UIFlowController uiFlowController;

    [Header("Result UI")]
    public RawImage resultPreviewImage;
    public TMP_Text resultStatusText;
    public Button retakeButton;
    public Button addLevelButton;
    public Button startGameButton;

    private bool isProcessing;
    private bool lastUploadSuccessful;
    private string currentServerIP;
    private LevelPayload pendingLevelPayload;
    private readonly Queue<LevelPayload> stagedLevelPayloads = new Queue<LevelPayload>();

    private string GetSanitizedIP()
    {
        return currentServerIP?
            .Trim()
            .Replace(" ", string.Empty)
            .Replace("\n", string.Empty)
            .Replace("\r", string.Empty)
            .Replace("http://", string.Empty)
            .Replace("https://", string.Empty);
    }

    private string GetSanitizedPort()
    {
        return port?
            .Trim()
            .Replace(" ", string.Empty)
            .Replace("\n", string.Empty)
            .Replace("\r", string.Empty);
    }

    private string UploadUrl => $"http://{GetSanitizedIP()}:{GetSanitizedPort()}/upload";
    private string DownloadUrl => $"http://{GetSanitizedIP()}:{GetSanitizedPort()}/download";

    private void Start()
    {
        ResolveUIFlowController();

        currentServerIP = PlayerPrefs.GetString("SavedServerIP", defaultServerIP);
        currentServerIP = GetSanitizedIP();

        if (ipInputField != null)
        {
            ipInputField.text = currentServerIP;
            ipInputField.onEndEdit.AddListener(UpdateIP);
        }

        if (saveIPButton != null)
        {
            saveIPButton.onClick.AddListener(() => UpdateIP(ipInputField != null ? ipInputField.text : currentServerIP));
        }

        if (retakeButton != null)
        {
            retakeButton.onClick.AddListener(ReturnToUploadPanel);
        }

        if (addLevelButton != null)
        {
            addLevelButton.onClick.AddListener(AddLevelAndReturnToUpload);
        }

        if (startGameButton != null)
        {
            startGameButton.onClick.AddListener(StartGameFromResult);
        }

        ShowUploadPanel();

        Debug.Log($"[MobileCameraCapture] Active upload endpoint: {UploadUrl}");
    }

    private void Update()
    {
#if ENABLE_INPUT_SYSTEM
        if (Keyboard.current != null && Keyboard.current.cKey.wasPressedThisFrame && !isProcessing)
        {
            PickFromCamera();
        }
#endif
    }

    public void UpdateIP(string newIP)
    {
        if (string.IsNullOrEmpty(newIP))
        {
            return;
        }

        string sanitized = newIP
            .Trim()
            .Replace(" ", string.Empty)
            .Replace("\n", string.Empty)
            .Replace("\r", string.Empty);

        if (sanitized.StartsWith("http://"))
        {
            sanitized = sanitized.Replace("http://", string.Empty);
        }

        if (sanitized.StartsWith("https://"))
        {
            sanitized = sanitized.Replace("https://", string.Empty);
        }

        currentServerIP = sanitized;

        if (ipInputField != null)
        {
            ipInputField.text = currentServerIP;
        }

        PlayerPrefs.SetString("SavedServerIP", currentServerIP);
        PlayerPrefs.Save();
        Debug.Log($"[MobileCameraCapture] Saved server IP: {currentServerIP}");
    }

    private void ShowUploadPanel()
    {
        ResolveUIFlowController();
        if (uiFlowController != null)
        {
            uiFlowController.ShowCapture();
        }
    }

    private void ShowResultPanel(bool success, string statusText)
    {
        ResolveUIFlowController();
        if (uiFlowController != null)
        {
            uiFlowController.ShowResult();
        }

        if (resultStatusText != null)
        {
            resultStatusText.text = statusText;
        }

        if (!success && resultPreviewImage != null)
        {
            resultPreviewImage.texture = null;
        }

        if (addLevelButton != null)
        {
            addLevelButton.interactable = success && pendingLevelPayload != null;
        }

        if (startGameButton != null)
        {
            startGameButton.interactable = success && (pendingLevelPayload != null || stagedLevelPayloads.Count > 0);
        }
    }

    private void ShowControlPanel()
    {
        ResolveUIFlowController();
        if (uiFlowController != null)
        {
            uiFlowController.ShowGameplay();
        }
    }

    public void ReturnToUploadPanel()
    {
        pendingLevelPayload = null;
        ShowUploadPanel();
    }

    public void ResetCaptureFlow()
    {
        pendingLevelPayload = null;
        stagedLevelPayloads.Clear();
        isProcessing = false;
        lastUploadSuccessful = false;
        ShowUploadPanel();
    }

    private void ResolveUIFlowController()
    {
        if (uiFlowController != null)
        {
            return;
        }

        uiFlowController = FindFirstObjectByType<UIFlowController>(FindObjectsInactive.Include);
        if (uiFlowController == null)
        {
            uiFlowController = new GameObject("UIFlowController").AddComponent<UIFlowController>();
        }
    }

    public void AddLevelAndReturnToUpload()
    {
        AddPendingLevelToFlow();
        pendingLevelPayload = null;
        ShowUploadPanel();
    }

    public void StartGameFromResult()
    {
        AddPendingLevelToStagedLevels();
        pendingLevelPayload = null;
        if (FlushStagedLevelsToCoordinator())
        {
            ShowControlPanel();
        }
        else
        {
            ShowUploadPanel();
        }
    }

    private void AddPendingLevelToFlow()
    {
        if (pendingLevelPayload == null)
        {
            return;
        }

        if (levelCoordinator != null && levelCoordinator.HasLoadedInitialLevel)
        {
            levelCoordinator.EnqueueLevel(pendingLevelPayload);
            Debug.Log($"[MobileCameraCapture] Added payload to active game queue. Pending in coordinator = {levelCoordinator.PendingLevelCount}");
            return;
        }

        AddPendingLevelToStagedLevels();
    }

    private void AddPendingLevelToStagedLevels()
    {
        if (pendingLevelPayload == null)
        {
            return;
        }

        stagedLevelPayloads.Enqueue(pendingLevelPayload);
        Debug.Log($"[MobileCameraCapture] Staged generated level. Staged count = {stagedLevelPayloads.Count}");
    }

    private bool FlushStagedLevelsToCoordinator()
    {
        if (levelCoordinator == null)
        {
            Debug.LogWarning("[MobileCameraCapture] LevelCoordinator is not assigned, staged payloads were not enqueued.");
            return false;
        }

        if (stagedLevelPayloads.Count == 0)
        {
            Debug.LogWarning("[MobileCameraCapture] Start Game ignored because there are no staged payloads.");
            return false;
        }

        while (stagedLevelPayloads.Count > 0)
        {
            levelCoordinator.EnqueueLevel(stagedLevelPayloads.Dequeue());
        }

        Debug.Log($"[MobileCameraCapture] Started game flow. Coordinator pending count = {levelCoordinator.PendingLevelCount}");
        return true;
    }

    public void PickFromCamera()
    {
        if (isProcessing)
        {
            return;
        }

        NativeCamera.TakePicture(path =>
        {
            if (!string.IsNullOrEmpty(path))
            {
                StartCoroutine(ProcessSelectedFiles(new[] { path }));
            }
        });
    }

    public void PickFromGallery()
    {
        if (isProcessing)
        {
            return;
        }

        if (preferMultipleGallerySelection && NativeGallery.CanSelectMultipleFilesFromGallery())
        {
            PickMultipleFromGallery();
            return;
        }

        if (preferMultipleGallerySelection && !fallbackToSingleGallerySelection)
        {
            Debug.LogWarning("[MobileCameraCapture] Multi-select gallery is unavailable on this platform/device.");
            return;
        }

        PickSingleFromGallery();
    }

    public void PickSingleFromGallery()
    {
        if (isProcessing)
        {
            return;
        }

        NativeGallery.GetImageFromGallery(path =>
        {
            if (!string.IsNullOrEmpty(path))
            {
                StartCoroutine(ProcessSelectedFiles(new[] { path }));
            }
        });
    }

    public void PickMultipleFromGallery()
    {
        if (isProcessing)
        {
            return;
        }

        if (!NativeGallery.CanSelectMultipleFilesFromGallery())
        {
            if (fallbackToSingleGallerySelection)
            {
                PickSingleFromGallery();
            }
            else
            {
                Debug.LogWarning("[MobileCameraCapture] Multi-select gallery is unavailable on this platform/device.");
            }

            return;
        }

        NativeGallery.GetImagesFromGallery(paths =>
        {
            if (paths != null && paths.Length > 0)
            {
                StartCoroutine(ProcessSelectedFiles(paths));
            }
        });
    }

    private IEnumerator ProcessSelectedFiles(IReadOnlyList<string> paths)
    {
        if (paths == null || paths.Count == 0)
        {
            yield break;
        }

        isProcessing = true;

        int total = 0;
        for (int i = 0; i < paths.Count; i++)
        {
            if (!string.IsNullOrEmpty(paths[i]))
            {
                total++;
            }
        }

        if (total == 0)
        {
            isProcessing = false;
            yield break;
        }

        int current = 0;
        for (int i = 0; i < paths.Count; i++)
        {
            string path = paths[i];
            if (string.IsNullOrEmpty(path))
            {
                continue;
            }

            current++;
            yield return StartCoroutine(ProcessSingleSelectedFile(path, current, total));
        }

        isProcessing = false;
    }

    private IEnumerator ProcessSingleSelectedFile(string path, int index, int total)
    {
        Debug.Log($"[MobileCameraCapture] Processing image {index}/{total}: {path}");

        Texture2D original = NativeCamera.LoadImageAtPath(path, -1, false);
        if (original == null)
        {
            Debug.LogError($"[MobileCameraCapture] Failed to load image: {path}");
            yield break;
        }

        Texture2D resized = GetResizedTexture(original);
        byte[] jpgData = resized.EncodeToJPG(jpgQuality);

        Destroy(original);
        if (resized != original)
        {
            Destroy(resized);
        }

        yield return StartCoroutine(PostImageToServer(jpgData, index, total));

        if (lastUploadSuccessful)
        {
            yield return StartCoroutine(DownloadImageInternal(index, total));
        }
        else
        {
            pendingLevelPayload = null;
            ShowResultPanel(false, "Processing failed");
        }
    }

    private Texture2D GetResizedTexture(Texture2D source)
    {
        if (source == null)
        {
            return null;
        }

        int sourceWidth = source.width;
        int sourceHeight = source.height;

        int targetWidth;
        int targetHeight;

        if (sourceWidth >= sourceHeight)
        {
            targetWidth = targetLongSide;
            targetHeight = Mathf.RoundToInt((float)sourceHeight / sourceWidth * targetLongSide);
        }
        else
        {
            targetHeight = targetLongSide;
            targetWidth = Mathf.RoundToInt((float)sourceWidth / sourceHeight * targetLongSide);
        }

        Debug.Log($"[MobileCameraCapture] Resize {sourceWidth}x{sourceHeight} -> {targetWidth}x{targetHeight}");
        return Resize(source, targetWidth, targetHeight);
    }

    private Texture2D Resize(Texture2D source, int targetWidth, int targetHeight)
    {
        RenderTexture rt = RenderTexture.GetTemporary(
            targetWidth,
            targetHeight,
            0,
            RenderTextureFormat.Default,
            RenderTextureReadWrite.Linear);

        Graphics.Blit(source, rt);

        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D result = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);
        result.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        result.Apply();

        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(rt);
        return result;
    }

    private IEnumerator PostImageToServer(byte[] data, int index, int total)
    {
        WWWForm form = new WWWForm();
        form.AddBinaryData("photo", data, $"mobile_upload_{index:D2}.jpg", "image/jpeg");

        using (UnityWebRequest www = UnityWebRequest.Post(UploadUrl, form))
        {
            www.SetRequestHeader("X-API-KEY", apiKey);
            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[MobileCameraCapture] Upload failed for image {index}/{total}: {www.error}");
                lastUploadSuccessful = false;
            }
            else
            {
                Debug.Log($"[MobileCameraCapture] Upload succeeded for image {index}/{total}.");
                lastUploadSuccessful = true;
            }
        }
    }

    public void GetImageFromServer()
    {
        if (isProcessing)
        {
            return;
        }

        StartCoroutine(DownloadImageRoutine());
    }

    private IEnumerator DownloadImageRoutine()
    {
        isProcessing = true;
        yield return StartCoroutine(DownloadImageInternal(1, 1));
        isProcessing = false;
    }

    private IEnumerator DownloadImageInternal(int batchIndex, int batchTotal)
    {
        int attempts = 0;
        bool isDone = false;

        while (attempts < maxPollingAttempts && !isDone)
        {
            attempts++;
            Debug.Log($"[MobileCameraCapture] Poll {batchIndex}/{batchTotal}, attempt {attempts}/{maxPollingAttempts}: {DownloadUrl}");

            using (UnityWebRequest www = UnityWebRequest.Get(DownloadUrl))
            {
                www.SetRequestHeader("X-API-KEY", apiKey);
                yield return www.SendWebRequest();

                if (www.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"[MobileCameraCapture] Download failed for image {batchIndex}/{batchTotal}: {www.error}");
                }
                else
                {
                    ServerResponse response = null;

                    try
                    {
                        response = JsonConvert.DeserializeObject<ServerResponse>(www.downloadHandler.text);
                    }
                    catch (System.Exception ex)
                    {
                        Debug.LogError($"[MobileCameraCapture] Failed to parse server response: {ex.Message}");
                        isDone = true;
                    }

                    if (response != null)
                    {
                        if (response.status == "processing")
                        {
                            Debug.Log($"[MobileCameraCapture] Server still processing image {batchIndex}/{batchTotal}.");
                        }
                        else if (response.status == "success" || response.metadata != null)
                        {
                            PrepareServerResponse(response, batchIndex);
                            ShowResultPanel(true, "Level generated successfully");
                            isDone = true;
                        }
                        else
                        {
                            Debug.LogError($"[MobileCameraCapture] Server returned error status: {response.status}");
                            pendingLevelPayload = null;
                            ShowResultPanel(false, string.IsNullOrEmpty(response.message) ? "Processing failed" : response.message);
                            isDone = true;
                        }
                    }
                }
            }

            if (!isDone && attempts < maxPollingAttempts)
            {
                yield return new WaitForSeconds(pollingInterval);
            }
        }

        if (!isDone)
        {
            Debug.LogError($"[MobileCameraCapture] Polling timed out after {maxPollingAttempts} attempts for image {batchIndex}/{batchTotal}.");
            pendingLevelPayload = null;
            ShowResultPanel(false, "Processing failed");
        }
    }

    private void PrepareServerResponse(ServerResponse response, int batchIndex)
    {
        if (response == null)
        {
            return;
        }

        if (response.metadata != null)
        {
            string trackJson = JsonConvert.SerializeObject(response.metadata);
            string savePath = Path.Combine(Application.persistentDataPath, $"schema_{batchIndex:D2}.json");
            File.WriteAllText(savePath, trackJson);
            File.WriteAllText(Path.Combine(Application.persistentDataPath, "schema.json"), trackJson);
            Debug.Log($"[MobileCameraCapture] Saved metadata backup: {savePath}");
        }

        Texture2D previewTexture = DecodeBase64ToTexture(GetPreviewBase64(response));
        if (resultPreviewImage != null && previewTexture != null)
        {
            resultPreviewImage.texture = previewTexture;
        }

        pendingLevelPayload = new LevelPayload
        {
            TrackData = response.metadata,
            ForegroundTexture = DecodeBase64ToTexture(GetForegroundBase64(response)),
            GameplayTexture = DecodeBase64ToTexture(GetGameplayBase64(response)),
            BackgroundTexture = DecodeBase64ToTexture(GetBackgroundBase64(response))
        };
    }

    private string GetPreviewBase64(ServerResponse response)
    {
        return response.images != null && !string.IsNullOrEmpty(response.images.preview_base64)
            ? response.images.preview_base64
            : null;
    }

    private string GetForegroundBase64(ServerResponse response)
    {
        return response.images != null && !string.IsNullOrEmpty(response.images.foreground_base64)
            ? response.images.foreground_base64
            : response.foreground_base64;
    }

    private string GetGameplayBase64(ServerResponse response)
    {
        return response.images != null && !string.IsNullOrEmpty(response.images.gameplay_base64)
            ? response.images.gameplay_base64
            : response.gameplay_base64;
    }

    private string GetBackgroundBase64(ServerResponse response)
    {
        return response.images != null && !string.IsNullOrEmpty(response.images.background_base64)
            ? response.images.background_base64
            : response.background_base64;
    }

    private Texture2D DecodeBase64ToTexture(string base64)
    {
        if (string.IsNullOrEmpty(base64))
        {
            return null;
        }

        try
        {
            byte[] imageBytes = System.Convert.FromBase64String(base64);
            Texture2D tex = new Texture2D(2, 2);
            if (tex.LoadImage(imageBytes))
            {
                return tex;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[MobileCameraCapture] Error decoding base64 texture: {e.Message}");
        }

        return null;
    }
}
