import camelot
import os

path = r"C:\Projects\listing-lens-RAG-Test\input\RHP_Modified.pdf"
script_dir = os.path.dirname(os.path.abspath(__file__))

# Extract from page 8 with lattice
tables = camelot.read_pdf(path, pages="8", flavor="lattice")
print(f"Found {len(tables)} tables with lattice on page 8")

if tables:
    print("Table preview:")
    print(tables[0].df.head())
    print("\nParsing report:")
    print(tables[0].parsing_report)
    
    # Save as Excel directly
    xlsx_path = os.path.join(script_dir, "rhp_employee_benefits_table.xlsx")
    tables.export(xlsx_path, f='excel')
    
    print(f"\n✓ Saved: {xlsx_path}")
    print(f"Accuracy: {tables[0].parsing_report['accuracy']:.1%}")
else:
    print("No tables found with lattice on page 8")
