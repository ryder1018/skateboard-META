using System;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Editor smoke test for the photo -> server -> level flow.
/// Enters Play mode, uploads one photo through MobileCameraCapture with the chosen art style,
/// waits for the result panel, starts the game and saves screenshots plus a report to
/// Temp/PipelineSmokeTest/. Run it from Tools/SkateP, or drop a request file at
/// Temp/pipeline_smoke_test.json: {"photo":"C:/x.jpg","style":"comic","server":"127.0.0.1"}.
/// </summary>
[InitializeOnLoad]
public static class PipelineSmokeTest
{
    [Serializable]
    private class Request
    {
        public string photo;
        public string style = "original";
        public string server = "127.0.0.1";
        public float playSeconds = 10f;
        public bool exitPlayMode = true;
    }

    private enum Stage { Idle, WaitingForPlay, Starting, ReadyToUpload, Processing, ResultShown, Playing }

    private const string RequestPath = "Temp/pipeline_smoke_test.json";
    private const string RefreshRequestPath = "Temp/pipeline_smoke_refresh";
    private const string OutputDir = "Temp/PipelineSmokeTest";
    private const string PendingKey = "PipelineSmokeTest.Pending";
    private const float ProcessingTimeoutSeconds = 400f;

    private static Stage stage = Stage.Idle;
    private static Request request;
    private static MobileCameraCapture capture;
    private static double stageStartedAt;
    private static double testStartedAt;
    private static string previousServerIP;
    private static bool tookMidGameplayShot;
    private static readonly StringBuilder report = new StringBuilder();

    static PipelineSmokeTest()
    {
        EditorApplication.update -= Update;
        EditorApplication.update += Update;
        EditorApplication.playModeStateChanged -= OnPlayModeChanged;
        EditorApplication.playModeStateChanged += OnPlayModeChanged;
    }

    [MenuItem("Tools/SkateP/Pipeline Smoke Test...")]
    private static void RunFromMenu()
    {
        string photo = EditorUtility.OpenFilePanel("Photo to upload", "", "jpg,jpeg,png");
        if (string.IsNullOrEmpty(photo))
        {
            return;
        }

        Start(new Request { photo = photo, style = PlayerPrefs.GetString("SelectedArtStyle", "original") });
    }

    private static void Start(Request newRequest)
    {
        if (stage != Stage.Idle || EditorApplication.isPlayingOrWillChangePlaymode)
        {
            Debug.LogWarning("[PipelineSmokeTest] Exit Play mode before starting a smoke test.");
            return;
        }

        request = newRequest;
        report.Clear();
        Directory.CreateDirectory(OutputDir);
        foreach (string old in Directory.GetFiles(OutputDir))
        {
            File.Delete(old);
        }

        SessionState.SetString(PendingKey, JsonUtility.ToJson(request));
        // The scene's LocalLevelTester would otherwise load its baked levels first.
        SessionState.SetBool(LocalLevelTester.SkipAutoApplyKey, true);
        stage = Stage.WaitingForPlay;
        EditorApplication.EnterPlaymode();
    }

    private static void OnPlayModeChanged(PlayModeStateChange change)
    {
        if (change == PlayModeStateChange.EnteredPlayMode)
        {
            string pending = SessionState.GetString(PendingKey, string.Empty);
            if (string.IsNullOrEmpty(pending))
            {
                return;
            }

            // Entering Play mode reloads the domain, so the static state starts over here.
            SessionState.EraseString(PendingKey);
            request = JsonUtility.FromJson<Request>(pending);
            report.Clear();
            testStartedAt = EditorApplication.timeSinceStartup;
            Log($"Request: photo={request.photo} style={request.style} server={request.server}");
            Enter(Stage.Starting);
        }
        else if (change == PlayModeStateChange.ExitingPlayMode && stage != Stage.Idle)
        {
            Finish(false, "Play mode was stopped before the test finished.");
        }
    }

    private static void Update()
    {
        if (stage == Stage.Idle)
        {
            // Lets scripts edited outside Unity recompile while the Editor is in the background.
            if (!EditorApplication.isPlayingOrWillChangePlaymode && File.Exists(RefreshRequestPath))
            {
                File.Delete(RefreshRequestPath);
                AssetDatabase.Refresh();
                return;
            }

            if (!EditorApplication.isPlayingOrWillChangePlaymode && File.Exists(RequestPath))
            {
                string json;
                try
                {
                    json = File.ReadAllText(RequestPath);
                    File.Delete(RequestPath);
                }
                catch (IOException)
                {
                    return; // still being written; try again next tick
                }

                Start(JsonUtility.FromJson<Request>(json));
            }

            return;
        }

        if (!EditorApplication.isPlaying)
        {
            return;
        }

        double inStage = EditorApplication.timeSinceStartup - stageStartedAt;
        switch (stage)
        {
            case Stage.Starting:
                if (inStage < 2.0)
                {
                    return;
                }

                PrepareUpload();
                break;

            case Stage.ReadyToUpload:
                if (inStage < 1.0)
                {
                    return;
                }

                capture.ProcessFiles(new[] { request.photo });
                Log("Upload started.");
                Enter(Stage.Processing);
                break;

            case Stage.Processing:
                if (capture.IsProcessing)
                {
                    if (inStage > ProcessingTimeoutSeconds)
                    {
                        Finish(false, $"No result after {ProcessingTimeoutSeconds}s.");
                    }

                    return;
                }

                Log($"Processing finished in {inStage:F1}s, level ready = {capture.HasPendingLevel}");
                Log($"Result text: {(capture.resultStatusText != null ? capture.resultStatusText.text : "(none)")}");
                Screenshot("02_result");
                Enter(Stage.ResultShown);
                break;

            case Stage.ResultShown:
                if (inStage < 2.0)
                {
                    return;
                }

                if (!capture.HasPendingLevel)
                {
                    Finish(false, "Server did not return a playable level.");
                    return;
                }

                capture.StartGameFromResult();
                Log("Pressed Start Game.");
                tookMidGameplayShot = false;
                Enter(Stage.Playing);
                break;

            case Stage.Playing:
                if (!tookMidGameplayShot && inStage > request.playSeconds * 0.4)
                {
                    tookMidGameplayShot = true;
                    Screenshot("03_gameplay_a");
                }

                if (inStage >= request.playSeconds)
                {
                    Screenshot("04_gameplay_b");
                    CheckGeneratedLevelIsPlaying();
                }

                break;
        }
    }

    private static void CheckGeneratedLevelIsPlaying()
    {
        LevelCoordinator coordinator = UnityEngine.Object.FindFirstObjectByType<LevelCoordinator>();
        if (coordinator == null || !coordinator.HasLoadedInitialLevel)
        {
            Finish(false, "No level was loaded after Start Game.");
            return;
        }

        int index = coordinator.CurrentLevelIndex;
        LevelPayload current = index >= 0 && index < coordinator.CurrentRunLevels.Count
            ? coordinator.CurrentRunLevels[index]
            : null;
        TrackData track = current != null ? current.TrackData : null;
        string playingStyle = track != null && track.art_style != null ? track.art_style.name : "original";
        Log($"Playing level {coordinator.CurrentLevelNumber}/{coordinator.TotalLevelCount}: " +
            $"{(track != null && track.points != null ? track.points.Count : 0)} track points, art style = {playingStyle}, " +
            $"layers = {current?.BackgroundTexture?.width}x{current?.BackgroundTexture?.height}");

        bool styleMatches = request.style == "original" ? track != null && track.art_style == null : playingStyle == request.style;
        if (track == null || track.points == null || track.points.Count < 2)
        {
            Finish(false, "The level in play has no track from the server.");
        }
        else if (!styleMatches)
        {
            Finish(false, $"The level in play has art style '{playingStyle}', expected '{request.style}'.");
        }
        else
        {
            Finish(true, "Photo -> server -> level -> gameplay flow completed.");
        }
    }

    private static void PrepareUpload()
    {
        Application.runInBackground = true;
        capture = UnityEngine.Object.FindFirstObjectByType<MobileCameraCapture>(FindObjectsInactive.Include);
        if (capture == null)
        {
            Finish(false, "No MobileCameraCapture in the open scene.");
            return;
        }

        if (!File.Exists(request.photo))
        {
            Finish(false, $"Photo not found: {request.photo}");
            return;
        }

        previousServerIP = PlayerPrefs.GetString("SavedServerIP", capture.defaultServerIP);
        capture.UpdateIP(request.server);
        if (capture.ArtStyleSelector != null)
        {
            capture.ArtStyleSelector.Select(request.style);
            Log($"Art style selector: {capture.ArtStyleSelector.SelectedLabel} ({capture.ArtStyleSelector.SelectedStyle})");
        }
        else
        {
            Log("Art style selector missing; the server default style will be used.");
        }

        Screenshot("01_upload_panel");
        Enter(Stage.ReadyToUpload);
    }

    private static void Enter(Stage next)
    {
        stage = next;
        stageStartedAt = EditorApplication.timeSinceStartup;
    }

    private static void Screenshot(string name)
    {
        string path = Path.Combine(OutputDir, name + ".png");
        ScreenCapture.CaptureScreenshot(path);
        Log($"Screenshot: {path}");
    }

    private static void Log(string line)
    {
        string stamped = $"[{EditorApplication.timeSinceStartup - testStartedAt,7:F1}s] {line}";
        report.AppendLine(stamped);
        Debug.Log($"[PipelineSmokeTest] {line}");
    }

    private static void Finish(bool passed, string summary)
    {
        if (stage == Stage.Idle)
        {
            return;
        }

        stage = Stage.Idle;
        SessionState.EraseBool(LocalLevelTester.SkipAutoApplyKey);
        if (previousServerIP != null)
        {
            PlayerPrefs.SetString("SavedServerIP", previousServerIP);
            PlayerPrefs.Save();
            previousServerIP = null;
        }

        Log((passed ? "PASS: " : "FAIL: ") + summary);
        Directory.CreateDirectory(OutputDir);
        File.WriteAllText(Path.Combine(OutputDir, "report.txt"), report.ToString());

        if (request != null && request.exitPlayMode && EditorApplication.isPlaying)
        {
            // Let the last screenshot be written before leaving Play mode.
            EditorApplication.delayCall += () => EditorApplication.delayCall += EditorApplication.ExitPlaymode;
        }
    }
}
