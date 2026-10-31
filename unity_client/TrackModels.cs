using System.Collections.Generic;

[System.Serializable]
public class StageTrackData
{
    public int stage_index;
    public string semantic_label;
    public string track_color;
    public List<float[]> points;
}

[System.Serializable]
public class ArtStyleData
{
    public string name;
    public string label;
    public string model;
    public float edge_f1;
}

[System.Serializable]
public class LevelDesignData
{
    public string generator;
    public string mission;
    public string design_notes;
    public List<string> surfaces;
}

[System.Serializable]
public class TrackData
{
    public string version;
    public string timestamp;
    public string source_image;
    public float aspect_ratio;
    public string track_color;
    public List<float[]> points;
    public List<int[]> occluded_segments;
    public string terrain_type;
    public float friction_coefficient;
    public int stage_count;
    public List<StageTrackData> tracks;
    public LevelDesignData level;
    public ArtStyleData art_style;
}
