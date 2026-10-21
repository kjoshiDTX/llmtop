//! A small parser for the Prometheus text exposition format, which vLLM and
//! llama.cpp both serve on `/metrics`.
//!
//! Samples borrow from the response body instead of copying it: `Sample<'a>`
//! can't outlive the `&'a str` it was parsed from. Label values that contain
//! escapes need unescaping, so they are `Cow`: borrowed when the text can be
//! used as is, owned only when it had to be rewritten.

use std::borrow::Cow;

pub type Labels<'a> = Vec<(&'a str, Cow<'a, str>)>;

#[derive(Debug, PartialEq)]
pub struct Sample<'a> {
    pub name: &'a str,
    pub labels: Labels<'a>,
    pub value: f64,
}

impl Sample<'_> {
    pub fn label(&self, key: &str) -> Option<&str> {
        self.labels.iter().find(|(k, _)| *k == key).map(|(_, v)| v.as_ref())
    }
}

/// Parses every sample line and skips comments, blanks, and lines it can't
/// read. A scraper should tolerate one odd line rather than fail the poll.
pub fn parse(text: &str) -> Vec<Sample<'_>> {
    text.lines().filter_map(parse_line).collect()
}

fn parse_line(line: &str) -> Option<Sample<'_>> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    let name_end = line.find(|c: char| c == '{' || c.is_whitespace())?;
    let name = &line[..name_end];
    let mut rest = &line[name_end..];

    let mut labels = Vec::new();
    if let Some(after_brace) = rest.strip_prefix('{') {
        let (parsed, after) = parse_labels(after_brace)?;
        labels = parsed;
        rest = after;
    }

    // An optional timestamp may follow the value; we ignore it.
    let value = parse_value(rest.split_whitespace().next()?)?;
    Some(Sample { name, labels, value })
}

/// Parses `key="value",key2="value2"}` and returns the labels plus whatever
/// follows the closing brace.
fn parse_labels(mut s: &str) -> Option<(Labels<'_>, &str)> {
    let mut labels = Vec::new();
    loop {
        s = s.trim_start_matches([',', ' ']);
        if let Some(after) = s.strip_prefix('}') {
            return Some((labels, after));
        }
        let eq = s.find('=')?;
        let key = s[..eq].trim();
        let (value, after) = parse_quoted(s[eq + 1..].strip_prefix('"')?)?;
        labels.push((key, value));
        s = after;
    }
}

/// Reads up to the closing quote, handling `\"`, `\\` and `\n`.
fn parse_quoted(s: &str) -> Option<(Cow<'_, str>, &str)> {
    let mut chars = s.char_indices();
    let mut owned: Option<String> = None;
    while let Some((i, c)) = chars.next() {
        match c {
            '"' => {
                let value = match owned {
                    Some(o) => Cow::Owned(o),
                    None => Cow::Borrowed(&s[..i]),
                };
                return Some((value, &s[i + 1..]));
            }
            '\\' => {
                // First escape seen: copy what we have so far, then keep appending.
                let buf = owned.get_or_insert_with(|| s[..i].to_string());
                match chars.next()?.1 {
                    'n' => buf.push('\n'),
                    other => buf.push(other),
                }
            }
            _ => {
                if let Some(buf) = owned.as_mut() {
                    buf.push(c);
                }
            }
        }
    }
    None
}

fn parse_value(s: &str) -> Option<f64> {
    match s {
        "NaN" => Some(f64::NAN),
        "+Inf" => Some(f64::INFINITY),
        "-Inf" => Some(f64::NEG_INFINITY),
        _ => s.parse().ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_names_labels_and_values() {
        let text = r#"
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen2.5-7B"} 3.0
llamacpp:requests_processing 1
some_histogram_bucket{le="+Inf"} 7 1700000000000
"#;
        let samples = parse(text);
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].name, "vllm:num_requests_running");
        assert_eq!(samples[0].label("model_name"), Some("Qwen/Qwen2.5-7B"));
        assert_eq!(samples[0].value, 3.0);
        assert_eq!(samples[1].labels, vec![]);
        assert_eq!(samples[2].value, 7.0);
    }

    #[test]
    fn unescapes_label_values() {
        let s = parse_line(r#"m{a="x\"y,z}",b="line\nbreak"} 1"#).unwrap();
        assert_eq!(s.label("a"), Some("x\"y,z}"));
        assert_eq!(s.label("b"), Some("line\nbreak"));
        assert!(matches!(s.labels[0].1, Cow::Owned(_)));
    }

    #[test]
    fn borrows_when_no_escapes() {
        let s = parse_line(r#"m{a="plain"} 1"#).unwrap();
        assert!(matches!(s.labels[0].1, Cow::Borrowed("plain")));
    }

    #[test]
    fn handles_special_values_and_junk() {
        assert!(parse_line("m NaN").unwrap().value.is_nan());
        assert_eq!(parse_line("m -Inf").unwrap().value, f64::NEG_INFINITY);
        assert_eq!(parse_line("m{a=\"unterminated} 1"), None);
        assert_eq!(parse_line("m notanumber"), None);
    }
}
