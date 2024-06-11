import re


class SectionAwareSplitter:
    def __init__(self, text):
        self.text = text.replace('\n', ' ').replace('\t', ' ').replace('  ', '')
        self.sections = {
            "1. Identification": ["Identification", "Product Identifier", "Product Identification", "Section 1", "Product and company identification"],
            "2. Hazard(s) identification": ["Hazard Identification", "Hazards Identification", "Section 2"],
            "3. Composition/information on ingredients": ["Composition", "Ingredients", "Information on Ingredients", "Section 3"],
            "4. First-aid measures": ["First Aid", "First Aid Measures", "Section 4"],
            "5. Fire-fighting measures": ["Fire Fighting", "Fire Fighting Measures", "Section 5"],
            "6. Accidental release measures": ["Accidental Release", "Accidental Release Measures", "Section 6"],
            "7. Handling and storage": ["Handling", "Storage", "Handling and Storage", "Section 7"],
            "8. Exposure controls/personal protection": ["Exposure Controls", "Personal Protection",
                                                         "Exposure Controls/Personal Protection", "Section 8"],
            "9. Physical and chemical properties": ["Physical Properties", "Chemical Properties",
                                                    "Physical and Chemical Properties", "Section 9"],
            "10. Stability and reactivity": ["Stability", "Reactivity", "Stability and Reactivity", "Section 10"],
            "11. Toxicological information": ["Toxicological Information", "Toxicology", "Section 11"],
            "12. Ecological information": ["Ecological Information", "Ecology", "Section 12"],
            "13. Disposal considerations": ["Disposal", "Disposal Considerations", "Section 13"],
            "14. Transport information": ["Transport Information", "Transport", "Section 14"],
            "15. Regulatory information": ["Regulatory Information", "Regulations", "Section 15"],
            "16. Other information": ["Other Information", "Other", "Section 16"],
        }

    def split_text(self):
        section_patterns = {
            key: re.compile(r'\b(?:' + '|'.join([re.escape(variation) for variation in variations]) + r')\b', re.IGNORECASE)
            for key, variations in self.sections.items()
        }

        section_positions = []
        for section, pattern in section_patterns.items():
            for match in re.finditer(pattern, self.text):
                section_positions.append((match.start(), section))

        section_positions.sort()

        sections = {}
        for i, (start, section) in enumerate(section_positions):
            end = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(self.text)
            sections[section] = self.text[start:end].strip()

        return sections


if __name__ == '__main__':
    text = """
        SECTION 1 : IDENTIFICATION
SAFETY DATA SHEET
     Product identifier used on
the label:
Product Name: Product Code: UPC Number:
Vitrified Bonded WHEEL
Bonded Abrasives 66253249782
  Other means of identification:
Recommended use of the chemical and restrictions on use:
Product Use/Restriction: Abrasive Product.
Chemical manufacturer address and telephone number: United States
Canada
Saint-Gobain Canada, Inc.
28 Albert Street, W. Plattsville, ON N0J 1S0
www.Nortonabrasives.com 519-684-7441
508-795-5000
For emergencies in Canada, call CHEMTREC: 800-424-9300
     Manufacturer Name: Address:
Website:
General Phone Number:
Emergency phone number:
Emergency Phone Number:
CHEMTREC:
Saint-Gobain Abrasives, Inc.
1 New Bond Street Worcester, MA 01615
www.Nortonabrasives.com 800-551-4413
508-795-5000
For emergencies in the US, call CHEMTREC: 800-424-9300
    SECTION 2 : HAZARD(S) IDENTIFICATION
Classification of the chemical in accordance with CFR 1910.1200(d)(f):
   Signal Word:
GHS Class:
Hazard Statements: Precautionary Statements:
Not applicable.
Not classified as hazardous according to OSHA Hazard Communication Standard, 29 CFR 1910.1200 Not applicable.
Not applicable.
Hazards not otherwise classified that have been identified during the classification process:
 Route of Exposure: Eye:
Skin: Inhalation:
Eyes. Skin. Inhalation. Ingestion.
Causes eye irritation.
Causes skin irritation.
Prolonged or excessive inhalation may cause respiratory tract irritation.
   Vitrified Bonded WHEEL 66253249782 Revison Date: 04/19/2018 1 of 7

 Ingestion:
Chronic Health Effects:
Signs/Symptoms:
Target Organs:
Aggravation of Pre-Existing Conditions:
May be harmful if swallowed. May cause vomiting. Prolonged or repeated contact may cause skin irritation. Overexposure may cause headaches and dizziness. Eyes. Skin. Respiratory system. Digestive system.
None generally recognized.
   SECTION 3 : COMPOSITION/INFORMATION ON INGREDIENTS
  Mixtures:
Chemical Name CAS#
Aluminum Oxide, Non-fibrous 1344-28-1 Amorphous Silica, Fused 60676-86-0
Ingredient Percent
60 - 100 by weight
5 - 10 by weight
EC Num.
215-691-6
262-373-8
   Notes :
Actual grinding tests with wheels known to contain Crystalline Silica did not produce any detectable amount of respirable free Crystalline Silica.
   SECTION 4 : FIRST AID MEASURES
  Description of necessary measures:
Eye Contact:
Skin Contact: Inhalation: Ingestion:
Immediately flush eyes with plenty of water for at least 15 to 20 minutes. Ensure adequate flushing of the eyes by separating the eyelids with fingers. Remove contacts if present and easy to do. Continue rinsing. Get medical attention, if irritation or symptoms of overexposure persists.
Immediately wash skin with soap and plenty of water. Get medical attention if irritation develops or persists.
If inhaled, remove to fresh air. If not breathing, give artificial respiration or give oxygen by trained personnel. Seek immediate medical attention.
If swallowed, do NOT induce vomiting. Call a physician or poison control center immediately. Never give anything by mouth to an unconscious person.
 Most important symptoms/effects, acute and delayed: Other First Aid: Not applicable.
Indication of immediate medical attention and special treatment needed: Note to Physicians: Not applicable.
SECTION 5 : FIRE FIGHTING MEASURES
Suitable and unsuitable extinguishing media:
Suitable Extinguishing Media: Use alcohol resistant foam, carbon dioxide, dry chemical, or water fog or spray when fighting fires involving this material.
        Unsuitable extinguishing media: Not applicable.
  Vitrified Bonded WHEEL 66253249782 Revison Date: 04/19/2018 2 of 7
 
 Specific hazards arising from the chemical: Hazardous Combustion Not applicable.
Byproducts:
Unusual Fire Hazards: Not applicable.
Special protective equipment and precautions for fire-fighters:
  Protective Equipment: Fire Fighting Instructions:
NFPA Ratings: NFPA Health:
NFPA Flammability: NFPA Reactivity:
SECTION 6 : ACCIDENTAL RELEASE MEASURES
Personal precautions, protective equipment and emergency procedures:
Personal Precautions: Evacuate area and keep unnecessary and unprotected personnel from entering the spill area. Use proper personal protective equipment as listed in Section 8.
Environmental precautions:
Environmental Precautions: Avoid runoff into storm sewers, ditches, and waterways. Methods and materials for containment and cleaning up:
Spill Cleanup Measures: Not applicable.
Methods and materials for containment and cleaning up:
As in any fire, wear Self-Contained Breathing Apparatus (SCBA), MSHA/NIOSH (approved or equivalent) and full protective gear.
Not applicable.
0
0 1 0
  1 0
         Methods for containment: Methods for cleanup:
Reference to other sections:
Other Precautions:
Contain spills with an inert absorbent material such as soil or sand. Prevent from spreading by covering, diking or other means. Provide ventilation.
Clean up spills immediately observing precautions in the protective equipment section. Place into a suitable container for disposal. Provide ventilation. After removal, flush spill area with soap and water to remove trace residue.
Not applicable.
    SECTION 7 : HANDLING and STORAGE
Precautions for safe handling:
Handling: Use with adequate ventilation. Avoid breathing vapor and contact with eyes, skin and clothing. Hygiene Practices: Wash thoroughly after handling. Avoid contact with eyes and skin. Avoid inhaling vapor or mist. Conditions for safe storage, including any incompatibilities:
Storage: Store in a cool, dry, well ventilated area away from sources of heat, combustible materials, and incompatible substances. Keep container tightly closed when not in use.
       SECTION 8: EXPOSURE CONTROLS, PERSONAL PROTECTION
   Vitrified Bonded WHEEL 66253249782 Revison Date: 04/19/2018 3 of 7

   EXPOSURE GUIDELINES:
         Ingredient Guideline OSHA Guideline NIOSH Guideline ACGIH Quebec Canada Ontario Canada
     Aluminum Oxide, Non-fibrous
    PEL-TWA: 5 mg/m3 Respirable fraction (R) PEL-TWA: 15 mg/m3 Total particulate/dust (T)
        TLV-TWA: 10 mg/m3
    VEMP-TWA: 10 mg/m3 Total particulate/dust (T)
   OEL-TWAEV: 10 mg/m3 Total particulate/dust (T)
   Amorphous Silica, Fused
     OSHA PEL-TWA 0.1 mg/m3
      REL-TWA: 0.05 mg/m3 (Respirable)
      ACGIH TLV-TWA 0.1 mg/m3
      VEMP-TWA: 0.1 mg/m3 Respirable fraction (R)
     OEL-TWAEV: 0.1 mg/m3 Respirable fraction (R)
    Ingredient Alberta Canada Mexico British Columbia Canada
   Aluminum Oxide, Non-fibrous
     OEL-TWA: 10 mg/m3
      MPE-PPT: 0.1 mg/m3 Respirable fraction (R)
      OEL-TWA: 3 mg/m3 Respirable fraction (R) OEL-TWA: 10 mg/m3 OEL-TWA: 10 mg/m3 Total particulate/dust (T) OEL-STEL: 20 mg/m3 Total particulate/dust (T)
             Amorphous Silica, OEL-TWA: 0.1 mg/m3 MPE-PPT: 0.1 mg/m3 Fused Respirable fraction (R) Respirable fraction (R)
          Appropriate engineering controls:
Engineering Controls:
Individual protection measures:
Eye/Face Protection:
Skin Protection Description: Respiratory Protection:
Other Protective: PPE Pictograms:
Use appropriate engineering control such as process enclosures, local exhaust ventilation, or other engineering controls to control airborne levels below recommended exposure limits. Good general ventilation should be sufficient to control airborne levels. Where such systems are not effective wear suitable personal protective equipment, which performs satisfactorily and meets OSHA or other recognized standards. Consult with local procedures for selection, training, inspection and maintenance of the personal protective equipment.
Wear appropriate protective glasses or splash goggles as described by 29 CFR 1910.133, OSHA eye and face protection regulation, or the European standard EN 166.
Chemical-resistant gloves and chemical goggles, face-shield and synthetic apron or coveralls should be used to prevent contact with eyes, skin or clothing.
A NIOSH approved air-purifying respirator with an organic vapor cartridge or canister may be permissible under certain circumstances where airborne concentrations are expected to exceed exposure limits. Protection provided by air purifying respirators is limited. Use a positive pressure air supplied respirator if there is any potential for an uncontrolled release, exposure levels are not known, or any other circumstances where air purifying respirators may not provide adequate protection.
Facilities storing or utilizing this material should be equipped with an eyewash facility and a safety shower.
       SECTION 9 : PHYSICAL and CHEMICAL PROPERTIES
PHYSICAL AND CHEMICAL PROPERTIES:
   Physical State Appearance: Color:
Odor:
Odor Threshold:
Boiling Point: Melting Point: Density: Solubility:
Solid article. Not determined. Odorless.
Not determined. Not determined. Not determined. Not determined. Not determined.
   Vitrified Bonded WHEEL 66253249782 Revison Date: 04/19/2018 4 of 7

 Vapor Density: Vapor Pressure: Evaporation Rate: pH:
Viscosity:
Coefficient of Water/Oil Distribution:
Flammability:
Flash Point:
Lower Flammable/Explosive Limit: Upper Flammable/Explosive Limit: Auto Ignition Temperature: Explosive Properties:
VOC Content:
Not determined. Not determined. Not determined. Not determined. Not determined. Not determined.
Not determined.
None.
Not applicable.
Not applicable.
Not applicable.
Excessive dust accumulation could present a potential combustible dust hazard. Not determined.
   SECTION 10 : STABILITY and REACTIVITY
  Reactivity:
Reactivity:
Chemical Stability:
Chemical Stability:
Possibility of hazardous reactions:
Hazardous Polymerization:
Conditions To Avoid:
Conditions to Avoid:
Incompatible Materials:
Incompatible Materials:
Hazardous Decomposition Products:
Special Decomposition Products:
Not applicable.
Stable under normal temperatures and pressures.
Not reported.
Heat, flames, incompatible materials, and freezing or temperatures below 32 deg. F.
Oxidizing agents. Strong acids and alkalis.
Not applicable.
         SECTION 11 : TOXICOLOGICAL INFORMATION
TOXICOLOGICAL INFORMATION:
Acute Toxicity: This product has not been tested for its toxicity.
     Carcinogens:
         ACGIH NIOSH OSHA IARC NTP MEXICO
  Aluminum Oxide, Non-fibrous
    A4 Not Classifiable as a Human
Carcinogen
    No Data
     No Data
     No Data
    No Data
          A4 Not Classifiable as a Human
Carcinogen
  Amorphous Silica, Fused No Data NIOSH No Data No Data No Data No Data carcinogen
             Vitrified Bonded WHEEL 66253249782 Revison Date: 04/19/2018 5 of 7

 Aluminum Oxide, Non-fibrous : RTECS Number:
Inhalation:
Amorphous Silica, Fused :
RTECS Number: Inhalation:
BD1200000
Inhalation - Rat TCLo: 200 mg/m3/5H/28W (Intermittent) [Lungs, Thorax, or Respiration - Structural or functional change in trachea or bronchi; Lungs, Thorax, or Respiration - Chronic pulmonary edema; Related to Chronic Data - death] (RTECS)
"VV7328000"
Inhalation - Rat TCLo: 197 mg/m3/6H/26W (Intermittent) [Lungs, Thorax, !or Respiration - Changes in lung weight] (RTECS)
     SECTION 12 : ECOLOGICAL INFORMATION
Ecotoxicity:
Ecotoxicity: Please contact the phone number or address of the manufacturer listed in Section 1 for information on ecotoxicity.
SECTION 13 : DISPOSAL CONSIDERATIONS
Description of waste:
Waste Disposal: Consult with the US EPA Guidelines listed in 40 CFR Part 261.3 for the classifications of hazardous waste prior to disposal. Furthermore, consult with your state and local waste requirements or guidelines, if
applicable, to ensure compliance. Arrange disposal in accordance to the EPA and/or state and local guidelines.
            SECTION 14 : TRANSPORT INFORMATION
  UN number:
UN proper shipping name: Transport hazard class(es): Packing group: Environmental hazards: Special precautions for user:
Not regulated Not regulated Not regulated Not regulated Not regulated Not regulated
as hazardous material for transportation. as hazardous material for transportation. as hazardous material for transportation. as hazardous material for transportation. as hazardous material for transportation. as hazardous material for transportation.
   SECTION 15 : REGULATORY INFORMATION
Safety, health and environmental regulations specific for the product:
Inventory Status
            Japan ENCS EINECS Number South Korea KECL Australia AICS Canada DSL
            Aluminum Oxide, Non-fibrous (1) -23 262-373-8 KE-01012 Listed Listed
         Vitrified Bonded WHEEL 66253249782 Revison Date: 04/19/2018 6 of 7

       Amorphous Silica, Fused
262-373-8 Listed
              TSCA Inventory Status
            Aluminum Oxide, Non-fibrous Listed
            Amorphous Silica, Fused Listed
         Aluminum Oxide, Non-fibrous : Canada IDL:
Amorphous Silica, Fused : Canada IDL:
Aluminum Oxide, Non-fibrous :
EC Number:
Amorphous Silica, Fused :
EC Number:
State Right To Know
Identified under the Canadian Hazardous Products Act Ingredient Disclosure List: 0.1%.50(1298)
Identified under the Canadian Hazardous Products Act Ingredient Disclosure List: 0.1%.1404(1487)
215-691-6
262-373-8
           RI MN IL PA MA
            Aluminum Oxide, Non-fibrous Listed Listed No Data Listed Listed
            Amorphous Silica, Fused Listed Listed
                NJ
  Aluminum Oxide, Non-fibrous
    Listed: NJ Hazardous List; Substance Number: 2891
                         SECTION 16 : ADDITIONAL INFORMATION
  HMIS Ratings:
HMIS Health Hazard: HMIS Fire Hazard: HMIS Reactivity:
SDS Creation Date: SDS Revision Date: SDS Revision Notes: SDS Format:
1 1 0
April 19, 2018 April 19, 2018 GHS Update
Copyright© 1996-2018 Enviance. All Rights Reserved.
    Health Hazard
1
  Fire Hazard
 1
  Reactivity
 0
 Personal Protection
     Vitrified Bonded WHEEL 66253249782 Revison Date: 04/19/2018 7 of 7

    """

    sections = SectionAwareSplitter(text).split_text()

    # print(sections["1. Identification"])

    for i, section in enumerate(sections):
        print(f'Section {i+1} {section}: section {sections[section]}')
        print("\n\n-----------------------------------------------------------------\n\n")
    print(len(sections.values()))

