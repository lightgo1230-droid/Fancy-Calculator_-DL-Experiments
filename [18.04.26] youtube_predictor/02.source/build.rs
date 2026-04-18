fn main() {
    // Embed the Lightgo ICO into the Windows .exe resource table
    // so Windows Explorer and the taskbar show the correct icon.
    if std::env::var("CARGO_CFG_TARGET_OS").map_or(false, |os| os == "windows") {
        let ico = "resources/lightgo.ico";
        if std::path::Path::new(ico).exists() {
            let mut res = winres::WindowsResource::new();
            res.set_icon(ico);
            res.compile().expect("Failed to compile Windows resources");
        }
    }
}
