// build.rs — generates abstract "L" icon and embeds it into the Windows executable

fn write_u16(buf: &mut Vec<u8>, v: u16) { buf.extend_from_slice(&v.to_le_bytes()); }
fn write_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }

fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t.clamp(0.0, 1.0) }

/// Returns true if (col, row) falls inside the abstract "L" shape
/// Design is expressed in 32×32 space and scaled to any size
fn in_l(col: f32, row: f32, size: f32) -> bool {
    let s = 32.0 / size;
    let c = col * s;
    let r = row * s;
    let vert  = c >= 6.0 && c < 14.0 && r >= 3.0  && r < 27.0;
    let horiz = c >= 6.0 && c < 26.0 && r >= 20.0 && r < 27.0;
    vert || horiz
}

/// Build RGBA pixel buffer for one icon size
pub fn make_icon_rgba(size: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    let sz = size as f32;

    for row in 0..size {
        for col in 0..size {
            let r = row as f32;
            let c = col as f32;

            if in_l(c, r, sz) {
                // Gradient top→bottom: #64B4FF → #26C6DA
                let t   = r / sz;
                let red   = lerp(100.0, 38.0,  t) as u8;
                let green = lerp(180.0, 198.0, t) as u8;
                let blue  = lerp(255.0, 218.0, t) as u8;
                px.extend_from_slice(&[red, green, blue, 255]);
            } else {
                // Soft glow within 2 px of the L
                let mut closest = f32::MAX;
                'outer: for dr in -3i32..=3 {
                    for dc in -3i32..=3 {
                        if in_l(c + dc as f32, r + dr as f32, sz) {
                            let d = ((dr * dr + dc * dc) as f32).sqrt();
                            if d < closest { closest = d; }
                            if closest <= 1.2 { break 'outer; }
                        }
                    }
                }
                if closest <= 1.2 {
                    px.extend_from_slice(&[50, 100, 170, 210]);
                } else if closest <= 2.2 {
                    px.extend_from_slice(&[25, 45,  90,  130]);
                } else {
                    px.extend_from_slice(&[13, 17,  23,  255]); // background
                }
            }
        }
    }
    px
}

/// Wrap RGBA pixels into a BMP blob suitable for embedding in an ICO file
fn rgba_to_ico_bmp(size: u32, rgba: &[u8]) -> Vec<u8> {
    let mut bmp = Vec::new();
    let and_stride = (size + 31) / 32 * 4;

    // BITMAPINFOHEADER (40 bytes)
    write_u32(&mut bmp, 40);
    write_u32(&mut bmp, size);
    write_u32(&mut bmp, size * 2);  // biHeight doubled per ICO spec
    write_u16(&mut bmp, 1);         // biPlanes
    write_u16(&mut bmp, 32);        // biBitCount
    write_u32(&mut bmp, 0);         // biCompression = BI_RGB
    write_u32(&mut bmp, size * size * 4);
    write_u32(&mut bmp, 0); write_u32(&mut bmp, 0);
    write_u32(&mut bmp, 0); write_u32(&mut bmp, 0);

    // XOR (color) pixels — bottom-to-top, BGRA order
    for row in (0..size).rev() {
        for col in 0..size {
            let i = ((row * size + col) * 4) as usize;
            bmp.push(rgba[i + 2]); // B
            bmp.push(rgba[i + 1]); // G
            bmp.push(rgba[i]);     // R
            bmp.push(rgba[i + 3]); // A
        }
    }

    // AND mask — all zeros = fully opaque (alpha channel handles transparency)
    bmp.extend(vec![0u8; (and_stride * size) as usize]);
    bmp
}

/// Build a complete .ico binary containing 16×16, 32×32 and 48×48 images
fn build_ico() -> Vec<u8> {
    let sizes: &[u32] = &[16, 32, 48];
    let images: Vec<Vec<u8>> = sizes.iter()
        .map(|&s| rgba_to_ico_bmp(s, &make_icon_rgba(s)))
        .collect();

    let mut ico = Vec::new();

    // ICONDIR header
    write_u16(&mut ico, 0);                    // reserved
    write_u16(&mut ico, 1);                    // type: icon
    write_u16(&mut ico, sizes.len() as u16);

    // ICONDIRENTRY array — image data starts right after all entries
    let header_bytes = 6 + 16 * sizes.len();
    let mut offset   = header_bytes as u32;

    for (i, &sz) in sizes.iter().enumerate() {
        let dim = if sz >= 256 { 0u8 } else { sz as u8 };
        ico.push(dim);                          // bWidth
        ico.push(dim);                          // bHeight
        ico.push(0);                            // bColorCount
        ico.push(0);                            // bReserved
        write_u16(&mut ico, 1);                 // wPlanes
        write_u16(&mut ico, 32);                // wBitCount
        write_u32(&mut ico, images[i].len() as u32);
        write_u32(&mut ico, offset);
        offset += images[i].len() as u32;
    }

    for img in &images { ico.extend_from_slice(img); }
    ico
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let ico_path = format!("{}\\app.ico", manifest);

    std::fs::write(&ico_path, build_ico())
        .expect("Failed to write app.ico");

    // Embed the icon into the Windows executable resource section
    let mut res = winres::WindowsResource::new();
    res.set_icon(&ico_path);
    res.compile().ok();  // Non-fatal on non-MSVC or cross-compile
}
