import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def setup_plot_style():
    """Configure matplotlib style for better-looking plots"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10

def analyze_bbox_categories(df, output_dir):
    """Analyze distribution of bounding box categories (visual elements)"""

    print("\nBounding Box Category Analysis")
    
    # Filter to bbox rows only
    bbox_df = df[df['data_type'] == 'bbox'].copy()
    
    if len(bbox_df) == 0:
        print("No bounding box annotations found.")
        return
    
    # Count categories
    category_counts = bbox_df['bbox_category'].value_counts()
    print(f"\nTotal bounding boxes annotated: {len(bbox_df)}")
    print(f"\nVisual Element Type Distribution:")
    print(category_counts)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    category_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title('Distribution of Visual Element Types (Bounding Box Categories)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Visual Element Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'bbox_category_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: bbox_category_distribution.png")
    plt.close()
    
    # Analyze subcategories for each category
    print("\nSubcategory Breakdown by Category:")
    
    for category in category_counts.index:
        cat_df = bbox_df[bbox_df['bbox_category'] == category]
        subcat_counts = cat_df['bbox_subcategory'].value_counts()
        if len(subcat_counts) > 0:
            print(f"\n{category} ({len(cat_df)} total):")
            for subcat, count in subcat_counts.items():
                if pd.notna(subcat):
                    print(f"  - {subcat}: {count}")

def analyze_bbox_subcategories(df, output_dir):
    """Analyze distribution of bounding box subcategories"""
    
    print("\nBounding Box Subcategory Analysis")
    
    bbox_df = df[df['data_type'] == 'bbox'].copy()
    
    # Count subcategories
    subcat_counts = bbox_df['bbox_subcategory'].dropna().value_counts()
    
    if len(subcat_counts) == 0:
        print("No subcategory annotations found.")
        return
    
    print(f"\nSubcategory Distribution (top 20):")
    print(subcat_counts.head(20))
    
    # Create histogram for top subcategories
    fig, ax = plt.subplots(figsize=(14, 7))
    top_subcats = subcat_counts.head(20)
    top_subcats.plot(kind='barh', ax=ax, color='coral', edgecolor='black')
    ax.set_title('Top 20 Visual Element Subcategories', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Subcategory', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'bbox_subcategory_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: bbox_subcategory_distribution.png")
    plt.close()

def analyze_architectural_styles(df, output_dir):
    """Analyze distribution of architectural styles"""
    
    print("\nArchitectural Style Analysis")
    
    # Get unique buildings
    buildings_df = df[['building_id', 'architectural_style', 'architectural_style_pretty']].drop_duplicates()
    
    style_counts = buildings_df['architectural_style_pretty'].value_counts()
    print(f"\nNumber of unique buildings: {len(buildings_df)}")
    print(f"\nArchitectural Style Distribution:")
    print(style_counts)
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(style_counts.values, labels=style_counts.index, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 11})
    ax.set_title('Distribution of Architectural Styles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'architectural_style_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: architectural_style_distribution.png")
    plt.close()

def analyze_bbox_sizes(df, output_dir):
    """Analyze bounding box size distribution"""
    
    print("\nBounding Box Size Analysis")
    
    bbox_df = df[df['data_type'] == 'bbox'].copy()
    
    # Calculate bbox areas (as percentage of image)
    bbox_df['bbox_area_percent'] = bbox_df['bbox_width_percent'] * bbox_df['bbox_height_percent']
    
    print(f"\nBounding Box Size Statistics (% of image area):")
    print(f"  Mean area: {bbox_df['bbox_area_percent'].mean():.2f}%")
    print(f"  Median area: {bbox_df['bbox_area_percent'].median():.2f}%")
    print(f"  Min area: {bbox_df['bbox_area_percent'].min():.2f}%")
    print(f"  Max area: {bbox_df['bbox_area_percent'].max():.2f}%")
    
    # Create histogram of bbox sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Area distribution
    ax1.hist(bbox_df['bbox_area_percent'], bins=50, color='lightgreen', edgecolor='black')
    ax1.set_title('Distribution of Bounding Box Areas', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Area (% of image)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.axvline(bbox_df['bbox_area_percent'].mean(), color='red', 
                linestyle='--', label=f'Mean: {bbox_df["bbox_area_percent"].mean():.2f}%')
    ax1.legend()
    
    # Width vs Height scatter
    ax2.scatter(bbox_df['bbox_width_percent'], bbox_df['bbox_height_percent'], 
                alpha=0.5, s=30, color='purple')
    ax2.set_title('Bounding Box Dimensions', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Width (% of image)', fontsize=10)
    ax2.set_ylabel('Height (% of image)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bbox_size_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: bbox_size_analysis.png")
    plt.close()

def analyze_repeated_elements(df, output_dir):
    """Analyze whether visual elements are repeated"""
    
    print("\nRepeated Visual Elements Analysis")
    
    bbox_df = df[df['data_type'] == 'bbox'].copy()
    
    repeated_counts = bbox_df['bbox_is_repeated'].value_counts()
    print(f"\nRepeated Element Distribution:")
    print(repeated_counts)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    repeated_counts.plot(kind='bar', ax=ax, color=['lightcoral', 'lightblue'], 
                         edgecolor='black')
    ax.set_title('Are Visual Elements Repeated?', fontsize=14, fontweight='bold')
    ax.set_xlabel('Repeated Status', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'repeated_elements_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: repeated_elements_distribution.png")
    plt.close()

def analyze_annotations_per_building(df, output_dir):
    """Analyze annotation density per building"""

    print("\nAnnotations per Building Analysis")
    
    # Count bboxes per building
    bbox_counts = df[df['data_type'] == 'bbox'].groupby('building_name_pretty').size()
    
    print(f"\nBounding Boxes per Building:")
    print(f"  Mean: {bbox_counts.mean():.2f}")
    print(f"  Median: {bbox_counts.median():.2f}")
    print(f"  Min: {bbox_counts.min()}")
    print(f"  Max: {bbox_counts.max()}")
    
    print(f"\nTop 10 Buildings by Number of Annotations:")
    print(bbox_counts.sort_values(ascending=False).head(10))
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(bbox_counts, bins=20, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of Bounding Box Annotations per Building', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Bounding Boxes', fontsize=12)
    ax.set_ylabel('Number of Buildings', fontsize=12)
    ax.axvline(bbox_counts.mean(), color='red', linestyle='--', 
               label=f'Mean: {bbox_counts.mean():.2f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'annotations_per_building.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: annotations_per_building.png")
    plt.close()

def analyze_bboxes_per_image(df, output_dir):
    """Analyze how many bboxes per image"""
    
    print("\nBounding Boxes per Image Analysis")
    
    bbox_df = df[df['data_type'] == 'bbox'].copy()
    
    # Count bboxes per image
    bboxes_per_image = bbox_df.groupby(['building_id', 'image_index']).size()
    
    print(f"\nBounding Boxes per Image Statistics:")
    print(f"  Mean: {bboxes_per_image.mean():.2f}")
    print(f"  Median: {bboxes_per_image.median():.2f}")
    print(f"  Min: {bboxes_per_image.min()}")
    print(f"  Max: {bboxes_per_image.max()}")
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(bboxes_per_image, bins=range(int(bboxes_per_image.min()), 
                                          int(bboxes_per_image.max()) + 2), 
            color='plum', edgecolor='black', align='left')
    ax.set_title('Distribution of Bounding Boxes per Image', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Bounding Boxes', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.axvline(bboxes_per_image.mean(), color='red', linestyle='--', 
               label=f'Mean: {bboxes_per_image.mean():.2f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'bboxes_per_image.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: bboxes_per_image.png")
    plt.close()

def analyze_image_properties(df, output_dir):
    """Analyze image-level properties"""

    print("\nImage Properties Analysis")
    
    # Get unique images
    image_df = df[df['data_type'].isin(['image', 'bbox'])][
        ['building_id', 'image_index', 'image_has_text', 'image_unique_for_style']
    ].drop_duplicates(subset=['building_id', 'image_index'])
    
    print(f"\nTotal images: {len(image_df)}")
    
    # Has text distribution
    if 'image_has_text' in image_df.columns:
        has_text_counts = image_df['image_has_text'].value_counts()
        print(f"\nImages with Text:")
        print(has_text_counts)
    
    # Unique for style distribution
    if 'image_unique_for_style' in image_df.columns:
        unique_counts = image_df['image_unique_for_style'].value_counts()
        print(f"\nImages Unique for Architectural Style:")
        print(unique_counts)
    
    # Create side-by-side bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if 'image_has_text' in image_df.columns and len(has_text_counts) > 0:
        has_text_counts.plot(kind='bar', ax=ax1, color=['salmon', 'lightgreen'], 
                            edgecolor='black')
        ax1.set_title('Images with Text', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Has Text', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        ax1.tick_params(axis='x', rotation=0)
    
    if 'image_unique_for_style' in image_df.columns and len(unique_counts) > 0:
        unique_counts.plot(kind='bar', ax=ax2, color=['lightyellow', 'lightblue'], 
                          edgecolor='black')
        ax2.set_title('Images Unique for Style', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Unique for Style', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'image_properties.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: image_properties.png")
    plt.close()

def generate_summary_report(df, output_dir):
    """Generate a comprehensive summary report"""

    print("\nSummary Statistics Report")
    
    report_lines = []
    report_lines.append("DATASET SUMMARY REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("OVERALL STATISTICS:")
    report_lines.append(f"  Total rows in dataset: {len(df)}")
    report_lines.append(f"  Data type breakdown:")
    for dtype, count in df['data_type'].value_counts().items():
        report_lines.append(f"    - {dtype}: {count}")
    report_lines.append("")
    
    # Building statistics
    n_buildings = df['building_id'].nunique()
    report_lines.append("BUILDING-LEVEL STATISTICS:")
    report_lines.append(f"  Number of unique buildings: {n_buildings}")
    report_lines.append("")
    
    # Image statistics
    image_df = df[df['data_type'].isin(['image', 'bbox'])][
        ['building_id', 'image_index']
    ].drop_duplicates()
    n_images = len(image_df)
    report_lines.append("IMAGE-LEVEL STATISTICS:")
    report_lines.append(f"  Total images: {n_images}")
    report_lines.append(f"  Average images per building: {n_images / n_buildings:.2f}")
    report_lines.append("")
    
    # Bbox statistics
    bbox_df = df[df['data_type'] == 'bbox']
    n_bboxes = len(bbox_df)
    report_lines.append("BOUNDING BOX STATISTICS:")
    report_lines.append(f"  Total bounding boxes: {n_bboxes}")
    report_lines.append(f"  Average bboxes per building: {n_bboxes / n_buildings:.2f}")
    report_lines.append(f"  Average bboxes per image: {n_bboxes / n_images:.2f}")
    report_lines.append("")
    
    # Category statistics
    report_lines.append("VISUAL ELEMENT CATEGORIES:")
    for category, count in bbox_df['bbox_category'].value_counts().items():
        pct = (count / n_bboxes) * 100
        report_lines.append(f"  {category}: {count} ({pct:.1f}%)")
    report_lines.append("")
    
    report_lines.append("=" * 60)
    report_lines.append(f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save to file
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\nSaved: summary_report.txt")

def main():
    # Input file
    csv_file = '/home/kr3131/Mapping-Gothic-Scraper/benchmarking/v5/architecture_annotations.csv'
    
    # Create output directory for analysis results
    output_dir = Path('/mnt/swordfish-pool2/kavin/benchmarking/v5/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading data from: {csv_file}")
    
    # Load the CSV
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")
    
    # Setup plotting style
    setup_plot_style()
    
    # Run all analyses
    analyze_bbox_categories(df, output_dir)
    analyze_bbox_subcategories(df, output_dir)
    analyze_architectural_styles(df, output_dir)
    analyze_bbox_sizes(df, output_dir)
    analyze_repeated_elements(df, output_dir)
    analyze_annotations_per_building(df, output_dir)
    analyze_bboxes_per_image(df, output_dir)
    analyze_image_properties(df, output_dir)
    generate_summary_report(df, output_dir)
    
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()